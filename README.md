# MSINet: A Mask Structure Inference Network for Scene Text Image Super-Resolution



An official Pytorch implement of the paper "MSINet: A Mask Structure Inference Network for Scene Text Image Super-Resolution".




## Abstract

Understanding the structure of characters is crucial for recovering clear and readable high-resolution scene text images in Scene Text Image Super-Resolution (STISR). Recently, many existing STISR methods use the character structure information implicit in the recognition priors to guide the super-resolution process, helping generate more recognizable text images. However, the recognition priors obtained from low resolution are inaccurate, which easily misleads the super-resolution process. To address this problem, we draw inspiration from Masked Image Modeling (MIM) and propose the Mask Structure Inference Network (MSINet), which can generate scene text images with accurate character structures without recognition priors. To make STISR compatible with MIM, we also propose a Mask-and-Inference Paradigm (MIP), which consists of a mask image pre-training stage for character structure learning and a fine-tuning stage for character structure inference. In addition, a novel mask strategy named Text Confidence Mask (TCM) is proposed to avoid recovery errors by masking legible character regions. With MIP and TCM, MSINet impressively improves the clarity and readability of the degraded scene text images. Specifically, MSINet-B outperforms recent state-of-the-art methods by about +3.7\% on the TextZoom and average +3.6\% on six manually degraded scene text recognition datasets in recognition accuracy. 



Our code will be released after the paper is accepted
