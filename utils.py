import cv2

def str2bool(x): return x.lower() in ('true')

def norm(x): return (x - 0.5) / 0.5

def denorm(x): return x * 0.5 + 0.5

def tensor2numpy(x): return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x): return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def has_file_allowed_extension(filename, extensions):
    '''
    Checks if a file has an allowed extension.

    Args:
        filename (string):      Path to a file.
        extensions (string):    List of acceptable file extensions.

    Returns:
        bool:                   True if the filename ends with an acceptable image extension.
    '''    

    return any(filename.lower().endswith(ext) for ext in extensions)