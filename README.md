Try out the [Hunyuan Video Keyframe Lora](https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora) by running it on [Modal's Serverless AI cloud]. Just need to clone this repository and sign up for Modal  (if you haven't already).

Once you've signed up for Modal, just run:

    pip install -r requirements_local.txt

Then

    modal run app.py

You'll find the generated video stored on the Modal volume (in the "storage" area of your dashboard).

Inference is slow/expensive, but results aren't bad.

I haven't added an "if __name__=='__main__':" so you can either edit app.py if you want to run from command line or do what I do and modify prompt etc. in app.py directly. The settings are at the very bottom of the file in main().