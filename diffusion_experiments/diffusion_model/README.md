# Diffusion Models from Scratch in PyTorch

## Description
**A diffusion probabilistic model capable of image generation**
<br>
I implemented a scalable diffusion probabilistic model in PyTorch and trained it on the CIFAR-10 and CelebA datasets.
<br>

## Model Structure
**UNet Backbone** <br>
<pre>
 Conv2d                        Output head          Width of blocks:     Has attention: <br>
    ↓                               ↑ <br>
DownBlock---skip connections---> UpBlock            channels[0]          attention[0] <br>
    ↓                               ↑ <br>
    ⋮  ᅟᅟᅟᅟ       ᅟᅟᅟᅟ        ⋮                    ⋮                     ⋮ <br>
DownBlock---skip connections---> UpBlock            channels[-2]         attention[-2] <br>
    ↓                               ↑ <br>
DownBlock → ResidualBlock → …  → UpBlock            channels[-1]         attention[-1] <br>
</pre>
**UNet (inChannel, channels, attention, numRes, neckLength, timeEmbDim=64)** <br>
Conv2d at the beginning maps *inChannel* to *channels[0]*; *channels* is a list of integers, each corresponding to the width of DownBlock and UpBlock at each level. 
*attention* is a list of boolean values, each denoting whether there is self-attention at each level.
The number of ResidualBlock connecting the bottommost DownBlock and UpBlock is determined by *neckLength*. 
Sinusoidal time embedding is injected into every ResidualBlock throughout the U-Net. <br>

**ResidualBlock (inChannels, outChannels, timeEmbDim)** <br>
Two 3×3 convolutions with GroupNorm and SiLU activations, with a residual skip connection and additive time-embedding projection. <br>

**DownBlock (inChannels, blockChannels, numRes, timeEmbDim, attention=False, noDownsample=False)** <br>
A stack of *numRes* residual blocks followed by a strided convolution for downsampling (no downsample for the bottommost DownBlock), 
optionally including a self-attention layer. <br>

**UpBlock (inChannels, blockChannels, numRes, timeEmbDim, attention=False, noUpsample=False)** <br>
Bilinear upsampling (no upsample for the bottommost UpBlock) followed by concatenation with skip connections and *numRes* residual blocks, optionally with self-attention. <br>

## Training
I use a cosine noise schedule and sample a random diffusion timestep for each training image. This exposes the model to varying noise levels within each batch and allows it to learn denoising across the full diffusion process simultaneously. The model is trained to predict the injected noise at each timestep, using mean squared error (MSE) loss. Gradients from this loss are backpropagated to update the network parameters. Training is performed with batch size of 100, so 100 images are processed every gradient step. The model is trained unconditionally, so class labels are ignored. <br>

### Datasets
I first trained the model on the CIFAR-10 dataset to debug the network and add features, as it is smaller in dimensions. After I ensured the model trains properly and the sampling process is correct, I moved on to train on the CelebA dataset, which has smoother structures and higher perceptual quality. <br>
CIFAR-10 contains 10 object classes, including animals and vehicles. CelebA contains faces of celebrities. <br>
Below are some samples of training images for reference. <br>
<br>
**CIFAR-10:** <br>
<img width="500" height="47" alt="CIFAR10_examples" src="https://github.com/user-attachments/assets/b2c1e49a-3ff1-4356-b744-a786c6e674ce" /> <br>
**CelebA:** <br>
<img width="1000" height="94" alt="CelebA_examples" src="https://github.com/user-attachments/assets/109cb24f-9e2c-465d-bf92-fb91e170b8e9" /> <br>

For the model trained on CIFAR-10, I used *channels* = [64, 128, 256, 256] with 2 residual blocks in each UpBlock/DownBlock and self-attention in the last 2 layers. <br>
For the model trained on CelebA, I used *channels* = [64, 128, 256, 512, 512] with 3 residual blocks per UpBlock/DownBlock and self-attention in 3rd and 4th layers. <br>

### Model Progression
As the models trained for more steps and saw more images, the samples looked progressively less distorted and more recognizable. Below is a table detailing how samples compare after the model is trained for an increasing number of steps. All samples are taken using DDPM without EMA with the same seed for consistency in comparison. <br>
<br>
**CIFAR-10:** <br>
Steps <br>
50k:       <img width="500" height="47" alt="samples_50k_ddpm" src="https://github.com/user-attachments/assets/3bca7155-8872-4c18-99a0-a3e9dbda75a6" /> <br>
100k: <img width="500" height="47" alt="samples_100k_ddpm" src="https://github.com/user-attachments/assets/3220954b-2ac1-4e06-a15e-7a5b20d07859" /> <br>
250k: <img width="500" height="47" alt="samples_250k_ddpm" src="https://github.com/user-attachments/assets/c46e20b8-5d1a-4386-acd9-29ba452a6161" /> <br>
300k: <img width="500" height="47" alt="samples_300k_ddpm" src="https://github.com/user-attachments/assets/4ea0d029-9593-44a5-a326-b464e8172239" /> <br>
350k: <img width="500" height="47" alt="samples_350k_ddpm" src="https://github.com/user-attachments/assets/cfc531c1-c86d-4eec-bd6b-d8c9b3128856" /> <br>
<br>
**CelebA:** <br>
1k:          <img width="684" height="72" alt="samples_1k_ddpm_seed30" src="https://github.com/user-attachments/assets/06990df8-0bfe-47c5-8b88-aeb7a265a075" /> <br>
20k:     <img width="684" height="72" alt="samples_20k_ddpm_seed30" src="https://github.com/user-attachments/assets/a5ade5b1-ee77-4821-a621-1cde8175a3fc" /> <br>
40k:     <img width="684" height="72" alt="samples_40k_ddpm_seed30" src="https://github.com/user-attachments/assets/b57d0553-ee56-4649-8d2c-20f343f5b377" /> <br>
60k:     <img width="684" height="72" alt="samples_60k_ddpm_seed30" src="https://github.com/user-attachments/assets/3786f908-137f-4be6-a220-a1c5aca5cb19" /> <br>
80k:     <img width="684" height="72" alt="samples_80k_ddpm_seed30" src="https://github.com/user-attachments/assets/a4f3b7f7-120f-4ae0-97f9-483aaf5d5e32" /> <br>
100k: <img width="684" height="72" alt="samples_100k_ddpm_seed30" src="https://github.com/user-attachments/assets/7d6cb2ef-76c4-477d-b9b5-b6edfe6995cc" /> <br>
120k: <img width="684" height="72" alt="samples_120k_ddpm_seed30" src="https://github.com/user-attachments/assets/70886915-f0e1-48e2-b88c-44e67e7be423" /> <br>
140k: <img width="684" height="72" alt="samples_140k_ddpm_seed30" src="https://github.com/user-attachments/assets/6ec926c5-d192-4821-b706-2e47a44c81bb" /> <br>
160k: <img width="684" height="72" alt="samples_160k_ddpm_seed30" src="https://github.com/user-attachments/assets/0c30ae90-2976-4eef-8747-83b75e1bfb2a" /> <br>
180k: <img width="684" height="72" alt="samples_180k_ddpm_seed30" src="https://github.com/user-attachments/assets/f7eefdb5-3fbf-4b8b-8dff-313537683ba4" /> <br>

<br>
It's apparent how the model is denoising the same initial noise into progressively cleaner images as it is trained for more steps. This process is more easily observed in CelebA samples since faces have a consistent structure compared to the varying perspectives and objects of CIFAR-10 images. Samples are shown every 20k steps up to 180k, after which visual improvements become subtle. <br>

## Sampling

### DDPM vs DDIM
DDPM sampling follows the full learned reverse diffusion process, iteratively denoising the image across all timesteps from pure noise to a final sample. This stochastic process tends to preserve fine details and produce higher perceptual quality, but it is computationally expensive due to the large number of sampling steps. <br>
DDIM accelerates sampling by using a deterministic update rule that allows the diffusion trajectory to be traversed in far fewer steps. By skipping most intermediate timesteps, DDIM significantly reduces sampling time. However, this deterministic path can oversmooth high-frequency details, sometimes resulting in less realistic samples. <br>
Below is a comparison of samples generated from the same model checkpoint after 196k training steps using DDIM with 20 steps, DDIM with 100 steps, and DDPM. <br>
<br>
**DDIM (20 steps):**    <img width="684" height="140" alt="samples_196k_ddim20_seed30" src="https://github.com/user-attachments/assets/3cf07787-6f15-48d8-b446-5d474ead36da" /> <br>
**DDIM (100 steps):** 
<img width="684" height="140" alt="samples_196k_ddim100_seed30" src="https://github.com/user-attachments/assets/b423f0dd-9045-40ce-99f8-ece1aee0bf0b" /> <br>
**DDPM:** ᅠᅠᅠᅠ       <img width="684" height="140" alt="samples_196k_ddpm_seed30" src="https://github.com/user-attachments/assets/3fc358ee-f294-403a-89a3-bafa499c7dd4" /> <br>

After 50 to 100 steps, DDIM samples achieve a similar quality as DDPM samples, especially for sampling from models in later stages of training. Note how even when all of the above samples are taken using the same seed, DDPM and DDIM samples do not match because they map the same noise to different faces. 

### EMA
Exponential Moving Average (EMA) weights summarize model parameters from previous steps to maintain a smoothed version of the model. EMA weights have steadier optimization since raw weights are stochastically updated by randomized batches. As a result, EMA weights can lag behind raw weights in early training, but yield more stable and higher-quality diffusion samples at later training stages.

Below is a comparison between CelebA DDPM samples generated using raw vs EMA weights of the model after 196k steps. <br>
**Raw weights:**   <img width="684" height="72" alt="samples_196k_ddpm_seed40" src="https://github.com/user-attachments/assets/a35f364b-f4e5-409d-9f8a-c81d5e7320fd" /> <br>
**EMA weights:** <img width="684" height="72" alt="samples_196k_ddpm_ema_seed40" src="https://github.com/user-attachments/assets/88b12f2a-17c9-44ae-ae4c-21faf9eea44b" /> <br>

Below is a comparison between CIFAR-10 DDPM samples generated using raw vs EMA weights of the model after 100k steps. <br>
**Raw weights:**   <img width="500" height="47" alt="samples_100k_ddpm_seed40" src="https://github.com/user-attachments/assets/b1c99a86-b93b-4e45-8388-092521fd843d" /> <br>
**EMA weights:** <img width="500" height="47" alt="samples_100k_ddpm_ema_seed40" src="https://github.com/user-attachments/assets/e1abb103-ae78-42cd-b63d-ad6e04ae0fc8" /> <br>

## Results
**CelebA** <br>
100 DDPM samples with EMA after 229k steps: <br>
<img width="662" height="662" alt="100CelebA_samples_229k_ddpm_ema" src="https://github.com/user-attachments/assets/45b4163b-84f3-4792-b8e0-ca39a1b32a5d" /> <br>
<br>
**CIFAR-10:** <br>
100 DDPM samples with EMA after 350k steps: <br> 
<img width="500" height="500" alt="CIFAR10_samples_350k_ddpm_ema" src="https://github.com/user-attachments/assets/3207e00d-8f75-4e87-a2f5-6559e58652c4" /> <br>

From these final samples, it is clear that both models are successful at generating recognizable images, especially with the CelebA samples being nearly photorealistic. Some CelebA samples actually resemble celebrities present in the dataset. <br>
The CIFAR-10 samples are less distinguishable because of the low dimensionality of the photos and the model mixing similar objects, such as birds and planes, cats and dogs. However, considering that the model for CIFAR-10 is trained unconditionally without class labels, these samples still achieve a reasonable level of resemblance to training images. <br>

## Performance
For training, I used CUDA on an RTX A4000 GPU. <br>
The CIFAR-10 model takes 11 seconds to run 100 steps; the final 350k-step model took around 10 hours to train. <br>
The CelebA model takes 55 seconds to run 100 steps; the final 229k-step model took around 35 hours to train. <br> 
The CelebA model contains approximately 100 million parameters; storing EMA weights in FP16 results in a ~200 MB checkpoint. <br>

I recorded the time needed to generate 64 CIFAR-10 samples and the time needed to generate 16 CelebA samples using 
combinations of 100-step DDPM vs DDIM sampling and running on 4-core CPU vs CUDA on A4000 in the table below. <br>

|ᅟᅟ| CIFAR-10 | CelebA|
|:---|:----|:----|
|CPU|DDIM: 2m 18s <br> DDPM: 21m 45s|DDIM: 4m 4s <br> DDPM: 40m 6s|
|CUDA|DDIM: 6s <br> DDPM: 38s|DDIM: 8s <br> DDPM: 50s|

## How to Sample
Download the final EMA weights for CelebA from this link: https://drive.google.com/file/d/17Extm3Zon0xu72yz5jbeTMJ5VoAyeRoC/view?usp=sharing and place it into the CelebAcheckpoints folder. <br>
Run the file SampleEMA.py to sample 64 CIFAR-10 and 16 CelebA images using DDIM, both using EMA weights of their respective final models (takes ~7 minutes using CPU). <br>
To change the sampling method for CIFAR-10, specify the first argument of *sampleCIFAR10* to be "ddim" or "ddpm" for DDPM or DDIM sampling, respectively. <br>
Similarly, the sampling method for CelebA is determined by the first argument of *sampleCelebA*. <br>
The second argument of *sampleCIFAR10* and *sampleCelebA* determines the number of images in one row of the square grid of images generated. <br>
Refer to the table under the performance section for sampling time. <br>
It is recommended to use a CUDA-enabled GPU for reasonable runtimes when sampling CelebA using DDPM. <br>

## Limitations and Improvements
The CelebA model was trained on images cropped to 64×64 to keep training time and computational cost reasonable. While 64×64 samples are significantly more visually appealing than the 32×32 CIFAR-10 samples, this resolution still limits fine detail and overall sharpness. Training at higher resolutions, such as 128×128, would allow the model to capture more detailed facial structure and texture, resulting in clearer and more realistic samples. 
<br>
Despite strong overall performance, the final CelebA model occasionally produces errors in generation, such as the example depicted below: <br>
<img width="200" height="201" alt="image" src="https://github.com/user-attachments/assets/1b172278-a445-48c9-ab97-465722c1df97" /> <br>
These errors typically appear near the top of the image in the form of vertical smearing. This behavior is caused by a combination of dataset alignment bias, ambiguity in hair regions, and flaws amplified through multiple upsampling stages in the U-Net. Such errors are a known limitation of diffusion models trained on 64×64 CelebA, and they largely disappear when training at 128×128 resolution. <br>
<br>
In addition to resolution limitations, the CIFAR-10 samples occasionally exhibit mixing between visually similar object classes, such as birds and planes or cats and dogs. This occurs because the model is trained unconditionally, without access to class labels. A natural solution would be to incorporate class conditioning, which would allow the model to distinguish between object categories and generate more class-consistent samples. <br>

## References
* https://nathanbaileyw.medium.com/a-look-at-diffusion-models-79bd7e789964; U-Net, UpBlock, and DownBlock structure
* https://arxiv.org/pdf/2006.11239; DDPM sampling and training
* https://arxiv.org/abs/2010.02502; DDIM sampling
