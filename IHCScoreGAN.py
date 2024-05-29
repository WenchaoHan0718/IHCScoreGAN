import time, itertools
from dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from pathlib import Path
import scipy.ndimage.filters as filters

from modules import *
from utils import denorm, tensor2numpy, RGB2BGR

class IHCScoreGAN(object):
    ''' Defines the model class, where the train and test loop resides. '''

    def __init__(self, args):
        self.model_name = 'IHCScoreGAN'
        self.exp_name = args.exp_name
        self.phase = args.phase

        self.result_dir = Path(args.result_dir)
        self.input_dir = Path(args.input_dir)
        self.load_path = args.load_path
        self.trainA_dir = self.input_dir / 'trainA'
        self.trainB_dir = self.input_dir / 'trainB'
        self.testA_dir = self.input_dir / 'testA'
        assert all([query in os.listdir(self.input_dir) for query in ['trainA', 'testA', 'trainB']]), \
            'Input directory must contain the following directories with necessary images: "trainA", "trainB", and "testA".'

        self.iteration = args.iteration

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.save_images = args.save_images

        self.lr = args.lr
        self.decay_flag = args.decay_flag
        self.weight_decay = args.weight_decay

        self.neighborhood_size = args.neighborhood_size
        self.kp_threshold = args.kp_threshold

        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight

        self.n_res = args.n_res
        self.n_dis = args.n_dis
        self.ch = args.ch

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device

        self.norm = nn.InstanceNorm2d
        print(args)

    def build_model(self):
        """ Dataset and DataLoader initialization. """

        trainA_transform = trainB_transform = testA_transform = testB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        if self.phase=='train':
            self.trainA_dataset = ImageFolder(self.trainA_dir, trainA_transform)
            self.trainA_loader = DataLoader(self.trainA_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            self.trainB_dataset = ImageAndKeypointFolder(self.trainB_dir, trainB_transform)
            self.trainB_loader = DataLoader(self.trainB_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            print(f'trainA_dataset: {len(self.trainA_dataset)} samples.')
            print(f'trainB_dataset: {len(self.trainB_dataset)} samples.')

        elif self.phase=='test':
            self.testA_dataset  = ImageFolder(self.testA_dir, testA_transform, return_path=True)
            self.testA_loader  = DataLoader(self.testA_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            # self.testB_dataset  = ImageAndKeypointFolder(self.trainB_dir, trainB_transform, return_path=True)
            # self.testB_loader  = DataLoader(self.testB_dataset, batch_size=self.batch_size, shuffle=False)
            print(f'testA_dataset: {len(self.testA_dataset)} samples.')

       # Generator and Discriminator initialization.
        self.genA2B = KeypointGenerator(input_nc=3, output_nc=3, ngf=self.ch, norm_layer=self.norm).to(self.device)
        self.genB2A = UnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, norm_layer=self.norm).to(self.device)
        self.disGA  = NLayerDiscriminator(input_nc=3, ndf=self.ch, n_layers=3, norm_layer=self.norm).to(self.device)
        self.disGK  = NLayerDiscriminator(input_nc=3, ndf=self.ch, n_layers=3, norm_layer=self.norm).to(self.device)
        self.disGB  = NLayerDiscriminator(input_nc=3, ndf=self.ch, n_layers=3, norm_layer=self.norm).to(self.device)

        # Loss function initialization.
        self.L1_loss  = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.CE_loss  = nn.CrossEntropyLoss().to(self.device)

        # Model optimizer initialization.
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disGK.parameters()), lr=self.lr, betas=(0.5, 0.999))

    def train(self):
        ''' Training loop of the model. '''

        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disGK.train()

        # Load model and optionally apply a learning rate decay corresponding to the loaded iteration.
        if self.resume:
            assert self.load_path is not None, '--load_path argument must be set to a model weights path because the --resume flag is True.'
            self.load(self.load_path)
            print(" [*] Load SUCCESS")
            if self.decay_flag and start_iter > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        ### Begin training loop.
        print('training start !')
        start_time = time.time()

        start_iter = 0
        for step in range(start_iter, self.iteration + 1, self.batch_size):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            # Since we have two DataLoaders with separate indices, we wrap them in unique iterators.
            try:
                real_A, _ = next(trainA_iter)
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter)

            try:
                real_B, _, real_K = next(trainB_iter)
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _, real_K = next(trainB_iter)

            real_A, real_B, real_K = real_A.to(self.device), real_B.to(self.device), real_K.to(self.device)

            ### Optional, experimental code to detect and apply the source image's RoI mask (i.e., whitespace) 
            ### to the target image. Helps the model to learn how to ignore whitespace in the source image.
            # msk = real_A.sum(axis=1, keepdims=True)==3.0
            # real_B = torch.where(msk, -1, real_B)
            # real_K = torch.where(msk, -1, real_K)
            # G Forward
            
            fake_B_dict = self.genA2B(real_A) # Fake B
            fake_B = fake_B_dict['output']
            fake_K = fake_B_dict['keypoints']
            rec_A  = self.genB2A(fake_B) # Reconstructed A

            fake_A = self.genB2A(real_B) # Fake A
            rec_B_dict  = self.genA2B(fake_A) # Reconstructed B
            rec_B  = rec_B_dict['output']
            rec_K  = rec_B_dict['keypoints']

            ### Begin the generator update step.

            # Do not compute gradients for the discriminators in this step.
            for net in [self.disGA, self.disGB, self.disGK]:
                for param in net.parameters():
                    param.requires_grad = False

            self.G_optim.zero_grad()

            idt_A = self.genB2A(real_A) # Identity A

            idt_B_dict = self.genA2B(real_B) # Identity B
            idt_B = idt_B_dict['output']
            idt_K = idt_B_dict['keypoints']

            G_identity_loss_A = self.L1_loss(idt_A, real_A) * self.identity_weight * 0.5 # ||G_B(A) - A||
            G_identity_loss_B = self.L1_loss(idt_B, real_B) * self.identity_weight * 0.5 # ||G_A(B) - B||
            G_identity_loss_K = self.L1_loss(idt_K, real_K) * self.identity_weight * 0.5 # ||G_K(B) - K||

            fake_GA_logit = self.disGA(fake_B)
            G_ad_loss_GA  = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device)) * self.adv_weight # D_A(G_A(A))
            
            fake_GB_logit = self.disGB(fake_A)
            G_ad_loss_GB  = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device)) * self.adv_weight # D_B(G_B(B))

            fake_GK_logit = self.disGK(fake_K)
            G_ad_loss_GK  = self.MSE_loss(fake_GK_logit, torch.ones_like(fake_GK_logit).to(self.device)) * self.adv_weight

            G_recon_loss_A = self.L1_loss(rec_A, real_A) * self.cycle_weight # ||G_B(G_A(A)) - A||
            G_recon_loss_B = self.L1_loss(rec_B, real_B) * self.cycle_weight # ||G_A(G_B(B)) - B||
            G_recon_loss_K = self.L1_loss(rec_K, real_K) * self.cycle_weight # ||G_A(G_K(B)) - K||

            Generator_loss =  G_ad_loss_GA + G_recon_loss_A + G_identity_loss_A \
                            + G_ad_loss_GB + G_recon_loss_B + G_identity_loss_B \
                            + G_ad_loss_GK + G_recon_loss_K + G_identity_loss_K

            Generator_loss.backward()
            self.G_optim.step()

            ### Begin the discriminator update step.

            # Compute gradients for the discriminators in this step.
            for net in [self.disGA, self.disGB, self.disGK]:
                for param in net.parameters():
                    param.requires_grad = True

            self.D_optim.zero_grad()

            real_GA_logit = self.disGA(real_B)
            real_GB_logit = self.disGB(real_A)
            real_GK_logit = self.disGK(real_K)

            fake_GA_logit = self.disGA(fake_B.detach())
            fake_GB_logit = self.disGB(fake_A.detach())
            fake_GK_logit = self.disGK(fake_K.detach())

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device)) * self.adv_weight
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device)) * self.adv_weight
            D_ad_loss_GK = self.MSE_loss(real_GK_logit, torch.ones_like(real_GK_logit).to(self.device)) + self.MSE_loss(fake_GK_logit, torch.zeros_like(fake_GK_logit).to(self.device)) * self.adv_weight

            Discriminator_loss = D_ad_loss_GA + D_ad_loss_GK + D_ad_loss_GB
            Discriminator_loss.backward()
            self.D_optim.step()

            ### Debugging the model optimization.
            print("[%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss), end='\r')
            
            # Consider incrementing step relative to the batch size.
            # step += self.batch_size

            ### Model visualization checkpoint.
            if (step % self.print_freq) < self.batch_size and step != 0:
                train_sample_num = 5
                A2B = np.zeros((self.img_size * 8, 0, 3))
                B2A = np.zeros((self.img_size * 10, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disGK.eval()
                for _ in range(train_sample_num):
                    try:
                        real_A, _ = next(trainA_iter)
                        assert len(real_B)==self.batch_size, 'trainA_iter Empty'
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, _ = next(trainA_iter)

                    try:
                        real_B, _, real_K = next(trainB_iter)
                        assert len(real_B)==self.batch_size, 'trainB_iter Empty'
                    except:
                        trainB_iter = iter(self.trainB_loader)
                        real_B, _, real_K = next(trainB_iter)

                    real_A, real_B, real_K = real_A.to(self.device), real_B.to(self.device), real_K.to(self.device)

                    fake_B_dict = self.genA2B(real_A)
                    fake_B = fake_B_dict['output']
                    fake_K = fake_B_dict['keypoints']
                    fake_A = self.genB2A(real_B)

                    rec_A = self.genB2A(fake_B)
                    rec_B_dict = self.genA2B(fake_A)
                    rec_B = rec_B_dict['output']
                    rec_K = rec_B_dict['keypoints']

                    idt_A = self.genB2A(real_A)
                    idt_B_dict = self.genA2B(real_B)
                    idt_B = idt_B_dict['output']

                    # Extract predicted cell center points and perform a local maxima operation.
                    keypoints = denorm(fake_K[0,0].detach().cpu().numpy())
                    maxima = (keypoints == filters.maximum_filter(keypoints, self.neighborhood_size)) & (keypoints > self.kp_threshold)
                    coords = torch.stack(torch.meshgrid([torch.linspace(0, 255, 256), torch.linspace(0, 255, 256)], indexing='ij'), -1).to(torch.uint8)[maxima]

                    # Perform an argmax operation over the predicted cell types at the location of each cell center point.
                    classes = tensor2numpy(fake_K[0,1:])[maxima].reshape(-1, 2).argmax(axis=1)

                    img = np.ascontiguousarray(tensor2numpy(denorm(real_A[0])))
                    img = denorm(real_A[0]).permute(1, 2, 0).contiguous().detach().cpu().numpy()
                    for coord, type in zip(coords, classes):
                        coord = np.array([coord[1], coord[0]])
                        img = cv2.circle(img, coord, 6, [1, 0, 0] if type==0 else [0, 0, 1], thickness=-1)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                            RGB2BGR(tensor2numpy(denorm(idt_A[0]))),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B[0]))),
                                                            RGB2BGR(tensor2numpy(fake_K[0,0, None].repeat(3, 1, 1))),
                                                            RGB2BGR(tensor2numpy(fake_K[0,1, None].repeat(3, 1, 1))),
                                                            RGB2BGR(tensor2numpy(fake_K[0,2, None].repeat(3, 1, 1))),
                                                            RGB2BGR(img),
                                                            RGB2BGR(tensor2numpy(denorm(rec_A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                            RGB2BGR(tensor2numpy(real_K[0,0, None].repeat(3, 1, 1))),
                                                            RGB2BGR(tensor2numpy(real_K[0,1, None].repeat(3, 1, 1))),
                                                            RGB2BGR(tensor2numpy(real_K[0,2, None].repeat(3, 1, 1))),
                                                            RGB2BGR(tensor2numpy(denorm(idt_B[0]))),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A[0]))),
                                                            RGB2BGR(tensor2numpy(denorm(rec_B[0]))),
                                                            RGB2BGR(tensor2numpy(rec_K[0,0, None].repeat(3, 1, 1))),
                                                            RGB2BGR(tensor2numpy(rec_K[0,1, None].repeat(3, 1, 1))),
                                                            RGB2BGR(tensor2numpy(rec_K[0,2, None].repeat(3, 1, 1)))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.exp_name, 'img', f'A2B_{step}.png'), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.exp_name, 'img', f'B2A_{step}.png'), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disGK.train()

            ### Model weights checkpoint.
            if step % self.save_freq == 0 and step == 40000:
                self.save(os.path.join(self.result_dir, self.exp_name, 'model'), step)

    def save(self, dir, step):
        '''
        Saves the model weights to a file, so they can be loaded later.

        Args:
            dir (string):   Path to the model weights directory.
            step (string):  The step (iteration) at which the model weights were saved.
        '''

        # Only the A2B generator is needed for typical inferencing.
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        os.makedirs(dir, exist_ok=True)
        torch.save(params, os.path.join(dir, '_'.join((self.model_name, str(step) + '.pt'))))

    def load(self, dir):
        '''
        Loads the model weights for inference or fine-tuning.

        Args:
            dir (string):   Path to the model weights directory.
        '''
                
        params = torch.load(dir)

        # Only the A2B generator is needed for typical inferencing.
        self.genA2B.load_state_dict(params['genA2B'])

    def test(self, quant_report=True):
        ''' Testing loop of the model. '''

        try:
            self.load(self.load_path)
            print(" [*] Load SUCCESS")
        except:
            print(" [*] Load FAILURE")
            return
        
        if quant_report:
            log_dir = os.path.join(self.result_dir, self.exp_name, 'logs')
            log_file = os.path.join(log_dir, Path(self.load_path).stem + '.log')
            with open(log_file, 'w') as fp:
                fp.write('row,filepath,record_type,percent_positive_nuclei,3+_cells_%,2+_cells_%,1+_cells_%,0+_cells_%,3+_nuclei,2+_nuclei,1+_nuclei,0+_nuclei,total_nuclei,class\n')
        counts = {}
        
        self.genA2B.eval()

        pbar = tqdm(self.testA_loader, total=len(self.testA_loader))
        for (real_A, _, path_A) in pbar:
            stems = [Path(path).stem for path in path_A]
            sample_names = [stem.split('-')[0] for stem in stems]
            pbar.set_description(sample_names[0])

            real_A = real_A.to(self.device)
            fake_B_dict = self.genA2B(real_A)
            fake_B = fake_B_dict['output']
            fake_K = fake_B_dict['keypoints']

            counts, coords_list, types_list = quantify_keypoints(counts, fake_K, sample_names, return_coords=True, neighborhood_size=self.neighborhood_size, threshold=self.kp_threshold)

            # NOTE: Be careful if you have lots of images to process - this code will duplicate your dataset.
            if self.save_images:
                datas = [(np.ascontiguousarray(tensor2numpy(denorm(img))), np.ascontiguousarray(tensor2numpy(denorm(kp)))) for img, kp in zip(real_A, fake_K)]

                for data, stem, sample_name, coords, types in zip(datas, stems, sample_names, coords_list, types_list):
                    if (np.array(types)==0).sum()<3: continue
                    img, kp = data
                    for coord, type in zip(coords, types):
                        coord = np.array([coord[1], coord[0]])
                        img = cv2.circle(img, coord, 6, [1, 0, 0] if type==0 else [0, 0, 1], thickness=-1)

                    path_dir = os.path.join(self.result_dir, self.exp_name, '_'.join(('test', Path(self.load_path).stem)))
                    os.makedirs(path_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(path_dir, stem + '.png'), RGB2BGR(img) * 255.0)

                    # NOTE: Uncomment this block to save the center point predictions.
                    #       We can optionally stitch together center point predictions before running the
                    #       local maxima algorithm in order to resolve most double-counting issues at tile borders.
                    # path_dir = os.path.join(self.result_dir, self.exp_name, '_'.join(('test_cp', self.dataset, self.load_iteration + self.fold_str)))
                    # os.makedirs(path_dir, exist_ok=True)
                    # cv2.imwrite(os.path.join(path_dir, stem + '.png'), RGB2BGR(kp) * 255.0)

        # Example of saving out quantification results to a log file.
        if quant_report:
            with open(log_file, 'a') as fp:
                for sample_name in tqdm(counts.keys()):
                    sample_counts = counts[sample_name]
                    record = {
                        'row':sample_name,
                        'filepath': '',
                        'record_type': 'ihcseg',
                        'percent_positive_nuclei': round((sample_counts['positive']*100)/sample_counts['total'] if sample_counts['total'] else 0, 4),
                        '3+_cells_%': round((sample_counts['positive']/sample_counts['total'] if sample_counts['total'] else 0)*100, 4),
                        '2+_cells_%': 0,
                        '1+_cells_%': 0,
                        '0+_cells_%': round((sample_counts['negative']/sample_counts['total'] if sample_counts['total'] else 0)*100, 4),
                        '3+_nuclei': sample_counts['positive'],
                        '2+_nuclei': 0,
                        '1+_nuclei': 0,
                        '0+_nuclei': sample_counts['negative'],
                        'total_nuclei': sample_counts['total'],
                        'class': 'positive' if (sample_counts['positive']/sample_counts['total'] if sample_counts['total'] else 0) > .2 else 'negative'
                    }

                    fp.write(','.join([str(v) for v in record.values()]) + '\n')

def quantify_keypoints(counts, K, sample_names, neighborhood_size=30, threshold=0.8, return_coords=False):
    '''
    Quantifies the cells in K using a local maxima algorithm on the predicted cell center points, 
    followed by an argmax algorithm over the predicted cell types.

    Args:
        counts (dict):           A dictionary representing the total recorded counts per sample.
        K (tensor):              The predicted cell center points and cell types, output by the model's secondary branch.
        sample_names (list):     A list of the sample names corresponding to each sample, so that sample counts can be aggregated.
        neighborhood_size (int): The neighborhood size (in pixels) for the local maxima algorithm.
        threshold (float):       A local maxima prediction must fall above this value to be considered a valid center point.
        return_coords (bool):    Whether to return the coordinates of the detected center points. 
                                 For example, simple quantification does not need this, but visualization does.

    Returns:
        counts (dict):           A dictionary of the updated total recorded counts per sample.
        coords (list):           A 2D list containing center point coordinates of each cell.
        classes_list (list):     A list containing the predicted cell types of each cell.
    '''

    type2label = {0:'positive', 1:'negative'}

    for sample_name in sample_names: 
        if sample_name not in counts.keys(): counts[sample_name] = {'positive':0, 'negative':0, 'total':0}

    keypoints = K[:,0].detach().cpu().numpy()
    maximas = [(keypoint == filters.maximum_filter(keypoint, neighborhood_size)) & (keypoint > threshold) for keypoint in keypoints]
    classes_list = [tensor2numpy(k[1:])[maxima].reshape(-1, 2).argmax(axis=1) for k, maxima in zip(K, maximas)]
    for classes, sample_name in zip(classes_list, sample_names):
        for i in range(2): 
            classes_sum = (classes==i).sum()
            counts[sample_name][type2label[i]] += classes_sum
            counts[sample_name]['total'] += classes_sum

    if return_coords:
        coords = [torch.stack(torch.meshgrid([torch.linspace(0, 255, 256), torch.linspace(0, 255, 256)], indexing='ij'), -1).to(torch.uint8)[maxima] for maxima in maximas]
        return counts, coords, classes_list
    else: 
        return counts