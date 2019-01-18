
# coding: utf-8

# In[51]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma


# In[60]:


img_orig_path = '/nas/home/nbonettini/projects/prnu-anonymization-detector/dataset/original/Nikon_D200_0_14921.png'
img_sara_path = '/nas/home/nbonettini/projects/prnu-anonymization-detector/dataset/mandelli/Nikon_D200_0_14921.png'
img_kirc_path = '/nas/home/nbonettini/projects/prnu-anonymization-detector/dataset/kirchner/Nikon_D200_0_14921.png'

img_orig = np.array(Image.open(img_orig_path)) / 255.
img_sara = np.array(Image.open(img_sara_path)) / 255.
img_kirc = np.array(Image.open(img_kirc_path)) / 255.

# Wavelet
img_orig_den_wv = denoise_wavelet(img_orig, multichannel=True)
img_sara_den_wv = denoise_wavelet(img_sara, multichannel=True)
img_kirc_den_wv = denoise_wavelet(img_kirc, multichannel=True)


# In[61]:


img_orig_noise_wv = img_orig - img_orig_den_wv
img_sara_noise_wv = img_sara - img_sara_den_wv
img_kirc_noise_wv = img_kirc - img_kirc_den_wv
clim_wv = [np.min(np.concatenate([img_orig_noise_wv, img_sara_noise_wv, img_kirc_noise_wv])),
        np.max(np.concatenate([img_orig_noise_wv, img_sara_noise_wv, img_kirc_noise_wv]))]


# In[62]:


img_orig_noise_wv_fft = np.abs((np.fft.fft2(img_orig_noise_wv,(224*2,224*2),axes=[0,1])))[:224,:224]
img_sara_noise_wv_fft = np.abs((np.fft.fft2(img_sara_noise_wv,(224*2,224*2),axes=[0,1])))[:224,:224]
img_kirc_noise_wv_fft = np.abs((np.fft.fft2(img_kirc_noise_wv,(224*2,224*2),axes=[0,1])))[:224,:224]


# Quelle sopra ^^^ sono le features

# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')

clim_wv_fft = [np.min(np.concatenate([img_orig_noise_wv_fft, img_sara_noise_wv_fft, img_kirc_noise_wv_fft])),
        np.max(np.concatenate([img_orig_noise_wv_fft, img_sara_noise_wv_fft, img_kirc_noise_wv_fft]))]

plt.figure(figsize=(15,20))
plt.subplot(3,3,1)
plt.imshow(img_orig)
plt.title('Orig')

plt.subplot(3,3,4)
plt.imshow(img_sara)
plt.title('Sara')

plt.subplot(3,3,7)
plt.imshow(img_kirc)
plt.title('Kirchner')

plt.subplot(3,3,2)
plt.imshow(img_orig_noise_wv_fft.mean(axis=2), clim=clim_wv_fft)#, plt.colorbar()
plt.title('Orig Noise WV FFF')

plt.subplot(3,3,5)
plt.imshow(img_sara_noise_wv_fft.mean(axis=2),clim=clim_wv_fft)#, plt.colorbar()
plt.title('Sara Noise WV FFT')

plt.subplot(3,3,8)
plt.imshow(img_kirc_noise_wv_fft.mean(axis=2),clim=clim_wv_fft)#, plt.colorbar()
plt.title('Kirchner Noise WV FFT')

plt.subplot(3,3,3)
plt.imshow(img_orig_noise_wv.mean(axis=2), clim=clim_wv)#, plt.colorbar()
plt.title('Orig Noise WV')

plt.subplot(3,3,6)
plt.imshow(img_sara_noise_wv.mean(axis=2), clim=clim_wv)#, plt.colorbar()
plt.title('Sara Noise WV')

plt.subplot(3,3,9)
plt.imshow(img_kirc_noise_wv.mean(axis=2), clim=clim_wv)#, plt.colorbar()
plt.title('Kirchner Noise WV')

plt.show()

