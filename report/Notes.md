HARDWARE:
We will need colab pro+ to run training overnight on a A100, we will periodically save weights to google drive as there is still 24hour limit, need to get training asap
Modification: Most things we can achieve by a first layer swap, aka. create new first layer with increased channel count, copy old layer weights into new (could do comparison of not copying vs copying), then replace model.conv1 = newlayer, train model from there. Could add FiLM layers for segmentation map tokens. May modify Loss function to penalize variation from fed in depth map. Could add Input Dropout, certain depth predictions will be bad on some datasets (glass window on cars = bad depth map), but really good on others (teddybears), so dropout makes model less dependent on depth layer (more code = better, could compare model trained with/without)
Datasets and Metrics: PSNR/LPIPS/SSIM, Datasets idk, Could also use Chamfer Distance, which is error between predicted 3D shape and real shape (ShapeNet provides reference point cloud, which means we can add thai as a metric)

So looks like we won’t need to do much code change to SplatterImage (not ideal), we need to make a lot of code comparing different models for generating priors (comparing which are most accurate for the datasets we chose) to justify our choices. Make a lot of evaluation code for comparing model before/after, and graphing resulting images/files (could have github with resulting 3D Gaussian scenes that we link to in report? Allows reviewer to see models themselves)
Maybe improve report by doing ablation study, training multiple models separately (+depth, +depth+mask, +depth+mask+Plucker, etc.)

From the possible priors:
Planes
Shapes
Normals
Visibility Cues -> Ray directions (Plucker Coords)
Depth Map
Depth Edge Map
RGB Edge Map
Segmentation Mask
Based on research, we should NOT use planes, this is as certain things (for example teddybears in Teddybears CO3D) do not have dominant planes in their shape, albeit datasets like Chairs/cars do, the better solution is to use predicted normals, this is as they can describe both curved and flat surfaces. For normals we can generate them as normal maps, which means we no longer need FiLM layers.
Resources for normals: 
https://github.com/AdamKruschwitz/Normal-Map-Generation-Example/blob/main/script.js
https://medium.com/@a.j.kruschwitz/how-to-generate-a-normal-map-from-an-image-the-quick-and-dirty-way-36b73a18f1f1
Depth Map - Many models predict 0 to 1, not metric distance, apparently taking the 0-1 and converting it to a point cloud using camera information will produce better results (Need to link sources when I find them all again), the point cloud easily translates to base gaussian locations the model can use. This feature should help a lot with datasets like Cars/Chairs and Hydrants, as will help deal with thin structures (legs/wheels) which models can struggle with (again need to link sources). Also may need to modify loss function to add geometric consistency loss, aka penalize SplatterImage for not following our depth map
Segmentation Maks - DINO provides tokens describing segment, can feed into model via FiLM layers
Plucker Coords - Basically a map where each pixel is a vector representing the ray direction, CNN does not understand that the top left pixel is a ray in some specific angle, so feeding this info should be good, ray directions need to be calculated on virtual camera information (focal length, etc.).


Adjusting Model (New Channels):
1. Modify Input layer channel count
2. Copy old weights into new input layer
3. Zero all weights for new channels
4. Freeze Model except new layer
5. (Fine Tuning) Add LoRA Layers to the rest of model, (LoRA matrices stay unfrozen)
6. Train

Adding Multi Modal data:
1. Add FiLM layer between blocks and/or at bottleneck -> Requires creating anotehr model that takes input multimodal data, output alpha + beta that get applied to intermediate channels
OR
1. Add Cross Attention layer between blocks and/or at bottleneck -> Requires model than can upscale pixels to match multimodal embedding dimensions or vice versa
