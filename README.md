## WeCare

October is Breast Cancer Awareness Month. I have been constantly been working on Brain Tumor segmentation for good part of last year.

I recently read a global report : https://pubmed.ncbi.nlm.nih.gov/33538338/

Which pointed out that in 2020, Worldwide, an estimated 19.3 million new cancer cases (18.1 million excluding nonmelanoma skin cancer) and almost 10.0 million cancer deaths (9.9 million excluding nonmelanoma skin cancer) occurred in 2020. Female breast cancer has surpassed lung cancer as the most commonly diagnosed cancer, with an estimated 2.3 million new cases (11.7%), followed by lung (11.4%), colorectal (10.0 %), etc. With 6.9% of all deaths due to breast cancer.

That's 700000 breast cancer deaths worldwide in 2020 alone.

Covid has caused a lot of issues, one being delayed diagnosis of cancer patients. Patients had to stand in long queues to get any kind of reports and that itself caused a lot of lives to end. It made me realize how cutting down the time of diagnosis in any way can save lives. I have lost people to cancer so I wonder if we can help in breast cancer diagnosis using deep learning.

## What it does
I wanted to help with the work of radiologists by providing semi automatic tool that used deep learning to reduce their workload to get better inferences and medical report. I also wanted to spread awareness about breast cancer so I also provided details on self examination.

There are two deep learning models that have been employed in this project:

Deep learning UNet breast cancer segmentor
This segmentor provides a semi automatic method of segmenting a breast cancer from
breast ultrasound images, which provides the initial mask which can be improved upon by radiologist and saved. This can help better inferences
Deep learning CNN breast cancer classifier(classes being normal, benign and malignant)

## How we built it
Four main components on how I built of the project

- Models: Holds details regarding the models that was trained and other details. Segmentation Model is written in tensorflow(UNET) and used breast cancer ultrasound dataset on kaggle. Classification model is an CNN model which is also written in tensorflow and trained on same dataset. This was the first step.

- Model_Server: Flask server that serves the inference models using REST api. This was the second setup and was relatively easy.

There are two endpoints:

  - predict : for segmentation model inference purpose
  - classify: for classification model inference purpose

- Vectorization_server: NodeJs-Express server that converts Raster images to Vector Image svg path. These svg path are the masks that can can overlaid on the orginal tumor for annotation purpose. This was horribly difficult as no good tools available in Python. This was the third step

- Annotation_Client: Dash app that provides the front-end to work with the segmentation+classification server and allows for manual annotation after automatic annotation is done by inference model. Also classifies the tumor using the classification model.

## What we learned
- Using CNN and UNET deep learnings models.
- Creating Dash apps
- UI/UX
- Deployements of tensorflow models

## References
- https://www.youtube.com/watch?v=zo0MNjG3jPY
