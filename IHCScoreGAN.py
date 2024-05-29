import time, itertools
from dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from modules import *
from glob import glob
from tqdm import tqdm
from pathlib import Path

import pandas as pd

import scipy.ndimage.filters as filters

def denorm(x): return x * 0.5 + 0.5
def tensor2numpy(x): return x.detach().cpu().numpy().transpose(1,2,0)
def RGB2BGR(x): return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

class IHCScoreGAN(object):
    def __init__(self, args):
        self.model_name = 'IHCScoreGAN'
        self.exp_name = args.exp_name
        self.phase = args.phase

        self.result_dir = Path(args.result_dir)
        self.input_dir = Path(args.input_dir)
        self.resume_path = args.resume_path
        self.trainA_dir = self.input_dir / 'trainA'
        self.trainB_dir = self.input_dir / 'trainB'
        self.testA_dir = self.input_dir / 'testA'
        self.testB_dir = self.input_dir / 'testB'
        assert all([query in os.listdir(self.input_dir) for query in ['trainA', 'testA', 'trainB', 'testB']]), \
            'Input directory must contain the following directories with input images: "trainA", "trainB", "testA" and "testB".'

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
        self.load_iteration = args.load_iteration

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
        """ DataLoader """
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

        """ Define Generator, Discriminator """
        self.genA2B = KeypointGenerator(input_nc=3, output_nc=3, ngf=self.ch, norm_layer=self.norm).to(self.device)
        self.genB2A = UnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, norm_layer=self.norm).to(self.device)
        self.disGA  = NLayerDiscriminator(input_nc=3, ndf=self.ch, n_layers=3, norm_layer=self.norm).to(self.device)
        self.disGK  = NLayerDiscriminator(input_nc=3, ndf=self.ch, n_layers=3, norm_layer=self.norm).to(self.device)
        self.disGB  = NLayerDiscriminator(input_nc=3, ndf=self.ch, n_layers=3, norm_layer=self.norm).to(self.device)

        """ Define Loss """
        self.L1_loss  = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.CE_loss  = nn.CrossEntropyLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disGK.parameters()), lr=self.lr, betas=(0.5, 0.999))

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disGK.train()

        if self.resume_path:
            self.load(self.resume_path)
            print(" [*] Load SUCCESS")
            if self.decay_flag and start_iter > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()

        start_iter = 0
        for step in range(start_iter, self.iteration + 1, self.batch_size):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

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

            # msk = real_A.sum(axis=1, keepdims=True)==3.0
            # real_B = torch.where(msk, -1, real_B)
            # real_K = torch.where(msk, -1, real_K)
            # G Forward
            
            fake_B_dict = self.genA2B(real_A) # fake B
            fake_B = fake_B_dict['output']
            fake_K = fake_B_dict['keypoints']
            rec_A  = self.genB2A(fake_B) # cycle A

            fake_A = self.genB2A(real_B) # fake A
            rec_B_dict  = self.genA2B(fake_A) # cycle B
            rec_B  = rec_B_dict['output']
            rec_K  = rec_B_dict['keypoints']

            # Update G
            for net in [self.disGA, self.disGB, self.disGK]:
                for param in net.parameters():
                    param.requires_grad = False

            self.G_optim.zero_grad()

            idt_A = self.genB2A(real_A) # identity

            idt_B_dict = self.genA2B(real_B) # identity
            idt_B = idt_B_dict['output']
            idt_K = idt_B_dict['keypoints']

            G_identity_loss_A = self.L1_loss(idt_A, real_A) * self.identity_weight * 0.5 # G_B should be identity if real_A is fed: ||G_B(A) - A||
            G_identity_loss_B = self.L1_loss(idt_B, real_B) * self.identity_weight * 0.5 # G_A should be identity if real_B is fed: ||G_A(B) - B||
            G_identity_loss_K = self.L1_loss(idt_K, real_K) * self.identity_weight * 0.5 # G_A should be identity if real_K is fed: ||G_K(B) - K||

            fake_GA_logit = self.disGA(fake_B)
            G_ad_loss_GA  = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device)) * self.adv_weight # GAN loss D_A(G_A(A))
            
            fake_GB_logit = self.disGB(fake_A)
            G_ad_loss_GB  = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device)) * self.adv_weight # GAN loss D_B(G_B(B))

            fake_GK_logit = self.disGK(fake_K)
            G_ad_loss_GK  = self.MSE_loss(fake_GK_logit, torch.ones_like(fake_GK_logit).to(self.device)) * self.adv_weight

            G_recon_loss_A = self.L1_loss(rec_A, real_A) * self.cycle_weight # Forward cycle loss || G_B(G_A(A)) - A||
            G_recon_loss_B = self.L1_loss(rec_B, real_B) * self.cycle_weight # Backward cycle loss || G_A(G_B(B)) - B||
            G_recon_loss_K = self.L1_loss(rec_K, real_K) * self.cycle_weight

            Generator_loss =  G_ad_loss_GA + G_recon_loss_A + G_identity_loss_A \
                            + G_ad_loss_GB + G_recon_loss_B + G_identity_loss_B \
                            + G_ad_loss_GK + G_recon_loss_K + G_identity_loss_K

            Generator_loss.backward()
            self.G_optim.step()

            # Update D
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

            print("[%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss), end='\r')
            # step += self.batch_size
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

                    keypoints = denorm(fake_K[0,0].detach().cpu().numpy())
                    maxima = (keypoints == filters.maximum_filter(keypoints, self.neighborhood_size)) & (keypoints > self.kp_threshold)
                    coords = torch.stack(torch.meshgrid([torch.linspace(0, 255, 256), torch.linspace(0, 255, 256)], indexing='ij'), -1).to(torch.uint8)[maxima]
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

            if step % self.save_freq == 0 and step == 40000:
                self.save(os.path.join(self.result_dir, self.exp_name, 'model'), step)

    def save(self, dir, step, fold=None):
        '''
        Input: 
            self
        '''
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disGK'] = self.disGB.state_dict()
        os.makedirs(dir, exist_ok=True)
        torch.save(params, os.path.join(dir, '_'.join((self.model_name, str(step) + '.pt'))))

    def load(self, dir, step=None):
        if step is not None: params = torch.load(os.path.join(dir, '_'.join((self.model_name, str(step) + '.pt'))))
        else: params = torch.load(dir)
        self.genA2B.load_state_dict(params['genA2B'])
        # self.genB2A.load_state_dict(params['genB2A'])
        # self.disGA.load_state_dict(params['disGA'])
        # self.disGB.load_state_dict(params['disGB'])
        # self.disGK.load_state_dict(params['disGK'])

    def test(self, quant_report=True):


        model_weights_list = glob(os.path.join(self.result_dir, self.exp_name, 'model', '*.pt'))
        if not len(model_weights_list) == 0:
            self.load(os.path.join(self.result_dir, self.exp_name, 'model'), self.load_iteration)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return
        
        if quant_report:
            log_dir = os.path.join(self.result_dir, self.exp_name, 'logs')
            log_file = os.path.join(log_dir, '_'.join(('quantification', self.dataset, str(self.load_iteration))) + '.log')
            os.makedirs(log_dir, exist_ok=True)
            with open(log_file, 'w') as fp:
                fp.write('row,filepath,record_type,percent_positive_nuclei,3+_cells_%,2+_cells_%,1+_cells_%,0+_cells_%,3+_nuclei,2+_nuclei,1+_nuclei,0+_nuclei,total_nuclei,class\n')
        counts = {}
        
        self.genA2B.eval()#, self.genB2A.eval()

        pbar = tqdm(self.testA_loader, total=len(self.testA_loader))
        for (real_A, _, path_A) in pbar:
            stems = [Path(path).stem for path in path_A]
            sample_names = [stem.split('-')[0] for stem in stems]
            pbar.set_description(sample_names[0])

            real_A = real_A.to(self.device)

            # import torchvision
            # real_A = torchvision.transforms.GaussianBlur(7, sigma=20)(real_A)

            fake_B_dict = self.genA2B(real_A)
            fake_B = fake_B_dict['output']
            fake_K = fake_B_dict['keypoints']

            counts, coords_list, types_list = quantify_keypoints(counts, fake_K, sample_names, return_coords=True, neighborhood_size=self.neighborhood_size, threshold=self.kp_threshold)

            if self.save_images:
                datas = [(np.ascontiguousarray(tensor2numpy(denorm(img))), np.ascontiguousarray(tensor2numpy(denorm(kp)))) for img, kp in zip(real_A, fake_K)]

                for data, stem, sample_name, coords, types in zip(datas, stems, sample_names, coords_list, types_list):
                    if (np.array(types)==0).sum()<3: continue
                    img, kp = data
                    for coord, type in zip(coords, types):
                        coord = np.array([coord[1], coord[0]])
                        img = cv2.circle(img, coord, 6, [1, 0, 0] if type==0 else [0, 0, 1], thickness=-1)

                    path_dir = os.path.join(self.result_dir, self.exp_name, '_'.join(('test', self.dataset, self.load_iteration)))
                    os.makedirs(path_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(path_dir, stem + '.png'), RGB2BGR(img) * 255.0)

                    path_dir = os.path.join(self.result_dir, self.exp_name, '_'.join(('test_cp', self.dataset, self.load_iteration + self.fold_str)))
                    os.makedirs(path_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(path_dir, stem + '.png'), RGB2BGR(kp) * 255.0)


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