CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jichu Mao
  * [LinkedIn](https://www.linkedin.com/in/jichu-mao-a3a980226/)
  *  [Personal Website](https://jichu.art/)
* Tested on: Windows 11,  i7-13700K @ 3.40 GHz, 32GB, RTX 4090 24GB

## Overview

Welcome to the GPU Path Tracer, a high-efficiency ray tracing engine built on CUDA for optimal performance.

This implementation, rooted in Physically-Based Rendering (PBR), integrates seamlessly with glTF assets for mesh and material processing. It accurately simulates light behavior, delivering effects such as soft shadows, caustics, and depth of field.

By harnessing the parallelism of NVIDIA's CUDA API, the project accelerates complex computations, reducing rendering time while maintaining visual fidelity. It explores advanced CUDA kernel optimization, memory coalescence, and spatial data structures. With 5,000 path traces per frame and up to 8 bounces per ray, we ensure fast convergence to achieve high-quality, noise-free renders.

## Highlights

| Duck-Doggy-Duck (800*800 5000spp)|
| ------------------------------------ |
| ![](./imgForReadMe/f2.png)         |

| The Dragon Beneath Starlit Skies (800*800 43200spp)|
| ------------------------------------ |
| ![](./imgForReadMe/f1.43203samp.png)         |

| Shanghai Nights (800*800 5000spp)|
| ------------------------------------ |
| ![](./imgForReadMe/f4.png)         |

| Stone Cat Guardian (800*800 5000spp)|
| ------------------------------------ |
| ![](./imgForReadMe/f3.png)         |

## Table of Contents
* [Highlights](#scenes)
* [Path Tracer Basics](#basic)
* Visual Features
	* [Material Shading(BSDFs)](#visual1)
	* [Stochastic Sampled Anti-Aliasing](#visual2)
	* [Physically-Based Depth of Field](#visual3)
	* [Arbitrary Mesh(glTF)Loading](#visual4)
	* [Texture Mapping](#visual5)
	* [Support for Environment Map](#visual6)
    * [Tone Mapping and Gamma Correction](#visual7)
* Performance Improvements
	* [Stream Compaction](#perf1)
    * [Material Sorting](#perf2)
	* [Bounding Box Culling for glTF](#perf3)

* [Bloopers](#bloopers)
* [References](#references)

## <a name="basic">Path Tracer Basics</a>

# Visual Features

## <a name="visual1"> Material Shading(BSDFs)</a>

## <a name="visual2"> Stochastic Sampled Anti-Aliasing</a>

## <a name="visual3"> Physically-Based Depth of Field</a>

This feature can be easily toggled ON and OFF using the preprocessor directive `DEPTH_OF_FIELD` in the `utilities.h` file.

## <a name="visual4"> Arbitrary Mesh(glTF) Loading</a>

## <a name="visual5"> Texture Mapping</a>

## <a name="visual6"> Support for Environment Map</a>

## <a name="visual7"> Tone Mapping and Gamma Correction</a>

In this project, I implemented the ACES/Reinhard Tonemapping methods, as well as Gamma Correction.

[Tone Mapping](https://en.wikipedia.org/wiki/Tone_mapping) is a process used to map high dynamic range (HDR) values to a lower dynamic range that output devices can display, preserving image details in highlights and shadows.

[ACES(Academy Color Encoding System)](https://en.wikipedia.org/wiki/Academy_Color_Encoding_System) is a color management framework designed to maintain color accuracy across different devices and formats, particularly in film production. It transforms high dynamic range and wide color gamut data into a manageable range for display or final output.

`ACES: color.rgb = (color.rgb * (2.51 * color.rgb + 0.03)) / (color.rgb * (2.43 * color.rgb + 0.59) + 0.14)`

[Reinhard Tone Mapping](https://64.github.io/tonemapping/) is a simple tone mapping technique that compresses HDR values to a lower range by scaling based on the brightness of each pixel. It avoids oversaturation in highlights.

`Reinhard: color.rgb = color.rgb / (1 + color.rgb)`

| <img src="imgForReadMe/notone.png" width=400> | <img src="imgForReadMe/reinhard.png" width=400> | <img src="imgForReadMe/aces.png" width=400>|
|:--:|:--:|:--:|
| No Tone Mapping | REINHARD_TONE_MAPPING ON|ACES_TONE_MAPPING ON|

[Gamma Correction](https://en.wikipedia.org/wiki/Gamma_correction)  adjusts the relationship between a pixel's input value and its displayed brightness, ensuring that the image appears correctly to the human eye, compensating for non-linear display behavior.

`Gamma Correction: color.rgb = pow(color.rgb, 1 / 2.2)`

| <img src="imgForReadMe/notone.png" > | <img src="imgForReadMe/gamma.png"> |
|:--:|:--:|
| `GAMMA_CORRECTION` OFF | `GAMMA_CORRECTION` ON|

These features can be easily toggled ON and OFF using the preprocessor directive `ACES_TONE_MAPPING` `REINHARD_TONE_MAPPING` `GAMMA_CORRECTION` in the `utilities.h` file.

# Performance Improvements

## <a name="perf1"> Stream Compaction</a>

## <a name="perf2"> Material Sorting</a>

## <a name="perf3"> Bounding Box Culling for glTF</a>


## <a name="bloopers">Bloopers!</a>
The Duck with Unevenly Applied "Cream"

 _* The sun brightness in the HDR environment map is too high, and the lack of multiple importance sampling in the code makes image convergence difficult *_
![](./imgForReadMe/blooper2.png)

The Agent Dog Traversing Across Dimensions

 _* Transformation matrix and camera setup misconfigured *_
![](./imgForReadMe/blooper1.png)




## <a name="references">References</a>
* Assets
    * [glTF Sample Models(Models)](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0)
    * [Sketchfab(Models)](https://sketchfab.com/)
    * [Poly Haven(HDR Maps& Models)](https://polyhaven.com/)
    * [HDRI-HAVEN(HDR Maps)](https://hdri-haven.com/)
* Implementations
    * Material Shading: [BSDFs](https://pbr-book.org/3ed-2018/Materials/BSDFs#BSDF), [Reflective Models](https://pbr-book.org/3ed-2018/Reflection_Models), [GPU Gems 3, Chapter 20](https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling)
    * Stochastic Sampled Anti-Aliasing: [Paul Bourke's Raytracing Notes](https://paulbourke.net/miscellaneous/raytracing/#:~:text=Stochastic%20Sampling&text=The%20method%20antialiases%20scenes%20through,a%20preset%20number%20of%20cells.)
    * Steam Compaction: [CIS 5650 Project 2 @UPenn](https://cis5650-fall-2024.github.io/) , [GPU Gems 3, Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
    * Depth of Field: [The Thin Lens Model and Depth of Field](https://www.pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models#TheThinLensModelandDepthofField)
    * glTF Mesh Loading: [Github Repo of tinygltf](https://github.com/syoyo/tinygltf/), [Official glTF Tutorials](https://github.com/KhronosGroup/glTF-Tutorials/)
    * Tone Mapping&Gamma Correciton: [The Evolution Theory of Tone Mapping](https://zhuanlan.zhihu.com/p/21983679)