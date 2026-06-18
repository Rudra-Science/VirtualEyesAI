// === FACE RECOGNITION VIP BOUNCER ===
// This script runs completely parallel to your YOLO app.

const FaceVIP = {
  isReady: false,
  faceMatcher: null,

  // 1. Define the roster of people and how many reference photos they have
  roster: [
    { name: 'Rudra', imageCount: 2 },
    { name: 'Rakesh', imageCount: 2 },
    { name: 'Deepa', imageCount: 2 }
  ],

  // 2. Boot up the models and memorize the faces
  async initialize() {
    try {
      console.log("👤 Booting up Facial Recognition models...");
      
      // Load the neural networks from your local /models folder
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models')
      ]);

      console.log("🧠 Models loaded. Memorizing VIP roster...");
      const labeledFaceDescriptors = await this.loadLabeledImages();
      
      // Create a matcher with a 60% confidence threshold
      this.faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
      this.isReady = true;
      console.log("✅ VIP Bouncer is online and ready.");

    } catch (error) {
      console.error("Facial Recognition failed to initialize:", error);
    }
  },

  // 3. Dig through the folders, look at the photos, and do the math
  async loadLabeledImages() {
    return Promise.all(
      this.roster.map(async (person) => {
        const descriptions = [];
        for (let i = 1; i <= person.imageCount; i++) {
          try {
            // Fetch the image from the local folder
            const img = await faceapi.fetchImage(`./labeled_images/${person.name}/${i}.jpg`);
            
            // Map the 128 points of the face
            const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
            
            if (detections) {
              descriptions.push(detections.descriptor);
            } else {
              console.warn(`Could not find a clear face in ${person.name}/${i}.jpg`);
            }
          } catch (e) {
            console.warn(`Missing reference image: ${person.name}/${i}.jpg`);
          }
        }
        return new faceapi.LabeledFaceDescriptors(person.name, descriptions);
      })
    );
  },

  // 4. The Live Intercept: Compare a live video frame against the roster
  async identifyPerson(videoElement) {
    if (!this.isReady) return "person"; // Default back to 'person' if not loaded

    // Scan the live video frame
    const detections = await faceapi.detectAllFaces(videoElement).withFaceLandmarks().withFaceDescriptors();
    
    if (detections.length === 0) return "person";

    // Match the live faces to our saved roster
    const results = detections.map(d => this.faceMatcher.findBestMatch(d.descriptor));
    
    // For simplicity, just return the name of the first face it recognizes
    if (results.length > 0 && results[0].label !== 'unknown') {
      return results[0].label; 
    }

    return "person"; // If it's a stranger, keep it as 'person'
  }
};

// Auto-start the initialization when this file loads
FaceVIP.initialize();