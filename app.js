const tf = require('@tensorflow/tfjs-node');
const cv = require('opencv4nodejs');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

// Load acne detection model
const loadModel = async () => {
  const model = await tf.loadLayersModel('file://./model_js/model.json');
  return model;
};

// Load image and detect faces
const detectFaces = async (imagePath, model) => {
  const image = await loadImage(imagePath);
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.width, image.height);

  const mat = cv.imread(imagePath);
  const faceCascade = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
  const gray = mat.bgrToGray();

  const faces = faceCascade.detectMultiScale(gray).objects;

  faces.forEach(async (faceRect) => {
    const { x, y, width, height } = faceRect;
    const faceMat = mat.getRegion(new cv.Rect(x, y, width, height));
    const resizedFace = faceMat.resize(224, 224);
    
    // Preprocess face for model input
    const faceTensor = tf.tensor3d(resizedFace.getDataAsArray(), [224, 224, 3])
      .expandDims(0)
      .toFloat()
      .div(127.5)
      .sub(1);

    // Make prediction
    const predictions = await model.predict(faceTensor).data();
    const [acne, withoutAcne] = predictions;
    const label = acne > withoutAcne ? "Acne" : "No Acne";
    const color = acne > withoutAcne ? new cv.Vec(0, 0, 255) : new cv.Vec(0, 255, 0);

    // Draw label and bounding box
    cv.drawRectangle(mat, new cv.Point2(x, y), new cv.Point2(x + width, y + height), color, 2);
    cv.putText(mat, `${label}: ${(Math.max(acne, withoutAcne) * 100).toFixed(2)}%`, 
               new cv.Point(x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
  });

  // Save or display the result
  cv.imwrite('output.jpg', mat);
};

// Run the face detection
(async () => {
  const model = await loadModel();
  await detectFaces('./image1.jpg', model);
  console.log('Detection completed and output saved as output.jpg');
})();
