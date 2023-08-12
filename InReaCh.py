import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

from typing import List, Union
import faiss 
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from FeatureDescriptors import *
from utils import *
from model import *
from mvtec_loader import *
import time

class InReaCh():
    def __init__(self, 
                 images: List[np.ndarray], 
                 model : torch.nn.Module,
                 assoc_depth: int = 10,
                 min_channel_length: int = 3,
                 max_channel_std: float = 5.0,
                 masks: List[np.ndarray] = None, 
                 quite: bool = False,
                 pos_embed_thresh: float = 1000,
                 pos_embed_weight: float = 5.0,
                 filter_size: float = 13,
                 **kwargs) -> None:
        
        self.quite = quite
        self.images = images
        self.masks = masks
        self.image_size = tuple(images[0].shape)
        self.model = model
        self.assoc_depth = assoc_depth
        self.filter_size = filter_size
        self.min_channel_length = min_channel_length
        self.max_channel_std = max_channel_std

        # Do positional embedding/alignment tests
        self.pos_embed_flag, self.images, self.masks, self.aligment_flag = positional_test_and_alignment(images, 
                                                                                                         threashold=pos_embed_thresh, 
                                                                                                         masks=self.masks, 
                                                                                                         align=False,
                                                                                                         quite=self.quite)
        self.pos_embed_weight = pos_embed_weight if self.pos_embed_flag else 0.

        # Do feature Extraction 
        self.fd_gen = Feautre_Descriptor(model=model, image_size=self.image_size, positional_embeddings = self.pos_embed_weight,   **kwargs)
        self.patches = self.fd_gen.generate_descriptors(self.images,quite=self.quite)  
        self.cpu_patches = self.patches.cpu().numpy()
        
        # If given masks track precision
        if not self.masks is None:
            self.patch_shape = (int(np.sqrt(self.cpu_patches.shape[2])),int(np.sqrt(self.cpu_patches.shape[2])))
            self.scale = masks[0].shape[0]//self.patch_shape[0] # This assumes square images...
            self.tp = 0
            self.fp = 0
            self.negatives = (np.count_nonzero(self.masks)//3)//self.scale
            self.positives = self.cpu_patches.shape[2]*self.cpu_patches.shape[0] - self.negatives
            self.max_label = np.max(np.array(self.masks))

        # Create Channels  
        self.gen_channels(self.quite)

    def gen_assoc(self, targets: torch.Tensor, 
                         sources: torch.Tensor, 
                         target_img_index: int, 
                         source_img_indexs: int):
        t_len = targets.size()[1]
        s_len = sources.size()[1]
        sources_zero_axis_min   = torch.from_numpy(np.ones(shape=(t_len))*np.inf).cuda()
        sources_zero_axis_index = torch.from_numpy(np.zeros(shape=(t_len))).cuda()
        targets_ones_axis_min   = torch.from_numpy(np.ones(shape=(s_len))*np.inf).cuda()
        targets_ones_axis_index = torch.from_numpy(np.zeros(shape=(s_len))).cuda()

        # Handle not having enough GPU memory to do everything in one big batch.
        aval_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        max_side = int(np.floor(np.sqrt(aval_mem//32)))
        for x in range(int(np.ceil(s_len/max_side))):
            for y in range(int(np.ceil(t_len/max_side))):

                distances = measure_distances(sources[:,x*max_side:min([(x+1)*max_side,s_len])],
                    targets[:,y*max_side:min([(y+1)*max_side,t_len])])

                mins, args = (torch.min(distances,axis=0))
                sources_zero_axis_index[y*max_side:min([(y+1)*max_side,t_len])] = torch.where(
                    sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])] >= mins,
                    args + x*max_side,
                    sources_zero_axis_index[y*max_side:min([(y+1)*max_side,t_len])] 
                )
                sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])] = torch.minimum(
                    sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])],
                    mins
                )

                mins, args = (torch.min(distances,axis=1))
                targets_ones_axis_index[x*max_side:min([(x+1)*max_side,s_len])] = torch.where(
                    targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])] >= mins,
                    args + y*max_side,
                    targets_ones_axis_index[x*max_side:min([(x+1)*max_side,s_len])]
                )
                targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])] = torch.minimum(
                    targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])],
                    mins
                )
        
        sources_indexs = sources_zero_axis_index.cpu().numpy().astype(int)
        targets_indexs = targets_ones_axis_index.cpu().numpy().astype(int)

        # Doing this on torch should speed this up
        assoc = np.ones((targets_indexs.shape[0],5))*np.inf
        for x in range(targets_indexs.shape[0]):
            if sources_indexs[targets_indexs[x]] == x:
                assoc[x] = [x,targets_indexs[x],targets_ones_axis_min[x].cpu().numpy(), target_img_index, source_img_indexs]
            else:
                assoc[x] = [np.inf,np.inf,targets_ones_axis_min[x].cpu().numpy(),np.inf,np.inf]

        return assoc

    def get_precision_recall(self):
        if not self.masks is None:
            return self.tp/(self.tp+self.fp), self.tp/self.positives
        else:
            return -1, -1

    def precision_recall(self, patches: List[list]):
        if not self.masks is None:
            for x in range(len(patches)):
                index = np.unravel_index(patches[x][2], shape=self.patch_shape)
                if np.average(self.masks[patches[x][1]][
                    index[0]*self.scale:(index[0]+1)*self.scale,
                    index[1]*self.scale:(index[1]+1)*self.scale,:]) == 0 : self.tp += 1
                else: self.fp += 1

    def gen_channels(self, quite: bool = False):
        # Collect assoc 
        assoc = np.ones((self.assoc_depth, self.patches.size(0), self.patches.size(2), 5))*np.inf
        for seed_index in tqdm.tqdm(range(self.assoc_depth), ncols=100, desc = 'Associate To Channels', disable=quite):  
            gpu_seeds = self.patches[seed_index].cuda()
            for compare_index in range(seed_index+1,self.patches.size(0)):
                assoc[seed_index,compare_index] = self.gen_assoc(gpu_seeds, self.patches[compare_index].cuda(), seed_index, compare_index)

        # Ensure each patch only associates to it's best candidate seed patch
        assoc = np.take_along_axis(assoc,np.expand_dims(assoc[:,:,:,2],axis=3).argmin(axis=0)[None],axis=0)[0]
        assoc = np.resize(assoc, (assoc.shape[0]*assoc.shape[1],assoc.shape[2]))
        # assoc -> [all_patches, [seed_p_index, img_p_index, distance, seed_image_index, img_image_index]]

        # Create Channels
        channels = {}
        for p_index in tqdm.tqdm(range(assoc.shape[0]), ncols=100, desc = 'Create Channels', disable=quite):
            if assoc[p_index,0] < np.inf:
                channel_name = str(int(assoc[p_index,0]))+'_'+str(int(assoc[p_index,3]))
                if channel_name in channels.keys():
                    channels[channel_name].append([self.cpu_patches[int(assoc[p_index,4]),:,int(assoc[p_index,1])], int(assoc[p_index,4]), int(assoc[p_index,1])])
                else:
                    channels[channel_name] = [[self.cpu_patches[int(assoc[p_index,3]),:,int(assoc[p_index,0])], int(assoc[p_index,3]), int(assoc[p_index,0])]]
                    channels[channel_name].append([self.cpu_patches[int(assoc[p_index,4]),:,int(assoc[p_index,1])], int(assoc[p_index,4]), int(assoc[p_index,1])])
                                                 #[         patch embedding                                       ,    img_image_index,    img_p_index]


        self.nn_object = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), self.patches.size(1), faiss.GpuIndexFlatConfig()) # deterministic brute force nn

        # Filter Channels 
        nominal_points = [] 
        for channel_name in tqdm.tqdm(list(channels.keys()), ncols=100, desc = 'Filter Channels', disable=quite):  
            if len(channels[channel_name])>self.min_channel_length:
                c_patches = [patch[0] for patch in channels[channel_name]]
                mean = np.mean(np.array(c_patches),axis=0)
                std = np.std(np.sqrt(np.sum(np.square(np.array(c_patches)-mean),axis=1)),axis=0) # Note we use spherical standard deviation
                new_centers = [center for center in channels[channel_name] if np.sqrt(np.sum(np.square(mean-center[0]))) < self.max_channel_std*std]
                c_patches = [patch[0] for patch in new_centers]
                if len(new_centers)>self.min_channel_length:
                    channels[channel_name] = new_centers
                    self.precision_recall(new_centers)
                    nominal_points += c_patches
                else:
                    del channels[channel_name]
            else:
                del channels[channel_name]
        self.nn_object.add(torch.from_numpy(np.array(nominal_points)))

    def predict(self, t_images: List[np.ndarray], 
                t_masks: List[np.ndarray] = None, 
                quite: bool = False): 
        if self.aligment_flag:
            t_images, t_masks = align_images(self.images[0], t_images, t_masks)

        start = time.time()
        t_patches =  self.fd_gen.generate_descriptors(t_images, quite=quite)
        
        scores = []
        for test_img_index in tqdm.tqdm(range(t_patches.size(0)), ncols=100, desc = 'Predicting On Images', disable=quite):
            dist, ind = self.nn_object.search(torch.permute(t_patches[test_img_index],(1,0)),1)
            dist = np.resize(dist[:,0], new_shape=(int(np.sqrt(dist.shape[0])),int(np.sqrt(dist.shape[0]))))
            dist = dist.repeat(t_images[0].shape[0]//dist.shape[0], axis=0).repeat(t_images[0].shape[0]//dist.shape[0], axis=1)
            scores.append(gaussian_filter(dist,self.filter_size))

        print('TIME TO COMPLETE ALL PREDICITONS')
        print('TIME TO COMPLETE all predictions', abs(start-time.time()))
        
        return scores, t_masks
         
    def test(self,  t_images: List[np.ndarray], 
                    t_masks: List[np.ndarray] = None, 
                    quite: bool = False):
        
        scores, t_masks = self.predict(t_images, t_masks=t_masks, quite=quite)

        t_masks = [(mask[:,:,0]/255.).astype(int) for mask in t_masks]

        img_scores = [np.max(score) for score in scores] 
        img_masks  = [np.max(mask) for mask in t_masks]

        scores  = np.array(scores).flatten()
        t_masks = np.array(t_masks).flatten()

        pxl_auroc = roc_auc_score(t_masks, scores)
        img_auroc = roc_auc_score(img_masks, img_scores)
        p, r = self.get_precision_recall()

        return pxl_auroc, img_auroc, p, r 


if __name__ == '__main__':
    # Test things with a basic config
    class_names = [ 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill','screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    return_nodes = {
        'layer1.0.relu_2': 'Level_1',
        'layer1.1.relu_2': 'Level_2',
        'layer1.2.relu_2': 'Level_3',
        'layer2.0.relu_2': 'Level_4',
        'layer2.1.relu_2': 'Level_5',
        'layer2.2.relu_2': 'Level_6',
        'layer2.3.relu_2': 'Level_7',
        'layer3.1.relu_2': 'Level_8',
        'layer3.2.relu_2': 'Level_9',
        'layer3.3.relu_2': 'Level_10',
        'layer3.4.relu_2': 'Level_11',
        'layer3.5.relu_2': 'Level_12',
        'layer4.0.relu_2': 'Level_13'
        }

    model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)

    average_pxl = []
    average_img = []
    average_percision = []
    average_recall = []

    for class_name in class_names:
        super_seed(112358)
        images, masks, corr_types = load_corrupted_data(class_name=class_name, 
                                                            data_dir='/home/declan/Desktop/Testing_Unsupervised_InReaCh_Competators/mvtec_anomaly_detection_10/',
                                                            num_corrupted=40)
        
        test_images, test_truths, test_class = load_testing_data(class_name=class_name, data_dir='/home/declan/Desktop/Testing_Unsupervised_InReaCh_Competators/mvtec_anomaly_detection_10/' )

        test_InReaCh = InReaCh(images=images, max_channel_std=5, model=model, masks=masks, quite=False)

        test_results = test_InReaCh.test(test_images, t_masks=test_truths, quite=False)
        average_pxl.append(test_results[0])
        average_img.append(test_results[1])
        average_percision.append(test_results[2])
        average_recall.append(test_results[3])

        print(class_name, test_results)

    print('averages', (np.average(average_pxl),np.average(average_img),np.average(average_percision),np.average(average_recall)))