**FaceMap**: Interpolating/transferring real-time user faces onto semantically-segmented StreetView buildings/roads/sky/... while users navigate around the world seen as them

<img src="/client/static/img/sample1.png" height="200"><img src="/client/static/img/sample2.png" height="200">
<img src="/client/static/img/sample3.png" height="200"><img src="/client/static/img/sample4.png" height="200">

High-level implementation description:
* Made use of public StreetView datasets for raw images with location coordinates to bypass API call limits
* Relayed webcam data over to a server along with user keystrokes, so that the player's face can be implanted onto the streetview while streetview navigation is calculated and managed
* Made use of fast neural style transfer, contour analysis of scenes, semantic segmentation (detectron2) to decompose scenes into components such as the sky, road, building, car, etc, and transfer weighted/interpolated masks of the users' warped faces in real-time
* Built a front-end with javascript, and connected it to flask python server to run all the image processing and load balancing

Installation instructions:
* Install `detectron2` from sources or through facebook's instructions
* Download [streets dataset](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)
* Run `client/game.py`

Demo video:

[<img src="/client/static/img/sample0.png" height="400">](https://www.youtube.com/watch?v=Jo9Km3TbcAg)
