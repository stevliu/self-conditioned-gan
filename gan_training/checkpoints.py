import os, pickle
import urllib
import torch
import numpy as np
from torch.utils import model_zoo


class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''

    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename, pretrained={}):
        '''Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        '''
        if 'model' in pretrained:
            filename = pretrained['model']
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename)

    def load_file(self, filename):
        '''Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        '''

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print('=> Loading checkpoint from local file...', filename)
            state_dict = torch.load(filename)
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            print('File not found', filename)
            raise FileNotFoundError

    def load_url(self, url):
        '''Load a module dictionary from url.

        Args:
            url (str): url to saved model
        '''
        print('=> Loading checkpoint from url...', url)
        state_dict = model_zoo.load_url(url, model_dir=self.checkpoint_dir, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        '''Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
    '''
        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {
            k: v
            for k, v in state_dict.items() if k not in self.module_dict
        }
        return scalars

    def load_clusterer(self, it, load_samples, pretrained={}):
        if 'clusterer' in pretrained:
            pretrained_file = os.path.join(self.checkpoint_dir, 'pretrained_clusterer.pkl')
            if not os.path.exists(pretrained_file):
                import cloudpickle as cp
                from urllib.request import urlopen
                print('Loading pretrained clusterer from', pretrained['clusterer'])
                clusterer = cp.load(urlopen(pretrained['clusterer'])) 
                print('Saving pretrained clusterer to', pretrained_file)
                with open(pretrained_file, 'wb') as f:
                    f.write(pickle.dumps(clusterer))
            else:
                with open(pretrained_file, 'rb') as f:
                    clusterer = pickle.load(f)
            return clusterer
        else:
            print('Loading clusterer:')
            with open(os.path.join(self.checkpoint_dir, f'clusterer{it}.pkl'), 'rb') as f:
                clusterer = pickle.load(f)

            if load_samples:
                print('Loading cluster samples:')
                with np.load(os.path.join(self.checkpoint_dir, 'cluster_samples.npz')) as f:
                    x = f['x']
                clusterer.x = torch.from_numpy(x)
            return clusterer

    def load_models(self, it, pretrained={}, load_samples=False):
        try:
            load_dict = self.load('model_%08d.pt' % it, pretrained)
            epoch_idx = load_dict.get('epoch_idx', -1)
        except Exception as e:  #models are not dataparallel modules
            print('Trying again to load w/o data parallel modules')
            try:
                for name, module in self.module_dict.items():
                    if isinstance(module, torch.nn.DataParallel):
                        self.module_dict[name] = module.module
                load_dict = self.load('model_%08d.pt' % it, pretrained)
                epoch_idx = load_dict.get('epoch_idx', -1)
            except FileNotFoundError as e:
                print(e)
                print("Models not found")
                it = epoch_idx = -1
        
        try:
            clusterer = self.load_clusterer(it, load_samples, pretrained)
        except FileNotFoundError as e:
            clusterer = None

        return it, epoch_idx, clusterer
    
    def save_clusterer(self, clusterer, it):
        with open(os.path.join(self.checkpoint_dir, f'clusterer{it}.pkl'), 'wb') as f:
            #hack: only save changing data
            x = clusterer.x
            clusterer.x = None
            pickle.dump(clusterer, f)
            clusterer.x = x

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
