const timeout_latency = 500; 
var socket_state = {keyup: ""};




// var bgImg;

// function preload() {
//     bgImg = loadImage("static/img/background.jpg");
// }

// function setup() {
//   createCanvas(1280, 680, WEBGL);
//   noStroke();
//   video = createCapture(VIDEO);
//   video.size(1280, 680);
//   video.position(0, 0);
//   video.hide();
// }

// function draw() {
//   bgImg = loadImage("host_image.jpg");
//   background(bgImg);

//   // "Fix" the coordinate system
//   let img = video.get( 0, 0, 1280, 680);


//   // Arbitrary demonstration of a 3d object drawn on top of the video
//   texture(img)
//   var socket = io.connect();
//   socket_state.img = img;
//   socket.emit('device_input', socket_state);
//   setTimeout(function(){
//       socket.disconnect();
//       socket_state = {img: ""};
//   }, timeout_latency); 


//   let rotationX = Math.cos(frameCount/2)/16
//   rotateX(rotationX)
//   let rotationY = Math.PI + Math.sin(frameCount/4)/8
//   rotateY(rotationY)
//   let translationY = 10*Math.sin(frameCount/4)
//   translate(0, translationY)
//   sphere(80); // 200
// }

// $(document).bind('keyup', function(keyUpEvent){
//   keyUpEvent.preventDefault();
//   $('#key').html(keyUpEvent.keyCode);
//   // let img = video.get( 0, 0, 1280, 680);

//   var socket = io.connect();
//   socket_state.keyup = keyUpEvent.keyCode;
//   // socket_state.img = img;
//   socket.emit('device_input', socket_state);
//   setTimeout(function(){
//       socket.disconnect();
//       socket_state = {keyup: ""};
//   }, timeout_latency); 

//   return false;
// });

$(document).bind('keyup', function(keyUpEvent){

  const canvas = document.querySelector('#canvas');
  canvas.style.display="none";
  const context = canvas.getContext('2d');

  // requestAnimationFrame(render);
  context.drawImage(video, 0, 0, window.innerWidth, window.innerHeight);
  const imgData = canvas.toDataURL("image/jpeg").split(';base64,')[1];

  keyUpEvent.preventDefault();
  $('#key').html(keyUpEvent.keyCode);
  // let img = video.get( 0, 0, 1280, 680);

  var socket = io.connect();
  socket_state.keyup = keyUpEvent.keyCode;
  socket_state.img_base64 =  imgData;
  // socket_state.img = img;
  socket.emit('device_input', socket_state);
  setTimeout(function(){
      socket.disconnect();
      socket_state = {keyup: "", img_base64: ""};
  }, timeout_latency); 

  return false;
});


// $("#dynamicImage").prop("src", "img/host_image.jpg?" + new Date().valueOf());

// $(function() {
//   var intervalMS = 5000; // 5 seconds
//   setInterval(function() {
//      $("#dynamicImage").prop("src", "img/host_image.jpg?" + +new Date());
//   }, intervalMS);
// });


initWebcamInput();
// render();

// function render(frameCount) {
//     // const canvas = document.getDocumentById("glCanvas")
//     const canvas = document.querySelector('#canvas');
//     canvas.style.display="none";
//     const context = canvas.getContext('2d');

//     requestAnimationFrame(render);
//     context.drawImage(video, 0, 0, window.innerWidth, window.innerHeight);
//     const imgData = canvas.toDataURL("image/jpeg").split(';base64,')[1];

//     var socket = io.connect();
//     socket_state.img_base64 =  imgData;
//     // socket_state.img = img;
//     socket.emit('device_input', socket_state);
//     setTimeout(function(){
//         socket.disconnect();
//         socket_state = {img_base64: ""};
//     }, timeout_latency); 

// };


function initWebcamInput() {

	if ( navigator.mediaDevices && navigator.mediaDevices.getUserMedia ) {

    navigator.mediaDevices.getUserMedia( { video: true } ).then( stream => {

        video.srcObject = stream;
        video.play();

    } );

   }

}


