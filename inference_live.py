import gc

from torchvision.transforms import ToTensor
import cv2
import torch

from model import MattingNetwork

# Choose the device to run the model on.
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
model = MattingNetwork('resnet50').eval().to(device)  # or "resnet50"
model.load_state_dict(torch.load('rvm_resnet50.pth'))

cap = cv2.VideoCapture(1)                                  # 

bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).to(device)  # Green background.
rec = [None] * 4                                           # Initial recurrent states.
downsample_ratio = 0.25                                    # Adjust based on your video.

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        src = ToTensor()(frame).unsqueeze(0)                           # To tensor.
        fgr, pha, *rec = model(src.to(device), *rec, downsample_ratio)  # Cycle the recurrent states.
        com = fgr * pha + bgr * (1 - pha)                              # Composite to green background. 

        # horizontal stack src and com for display.
        src = src.squeeze(0).permute(1, 2, 0).cpu().numpy()
        com = com.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame = cv2.hconcat([src, com])
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# empty the GPU cache.
if device.type == 'cuda':
    torch.cuda.empty_cache()
elif device.type == 'xla':
    torch.mps.empty_cache()

gc.collect()
