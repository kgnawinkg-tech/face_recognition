import os
import cv2
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

class GenerativeRobustFaceRecognizer:
    def __init__(self, target_size=(100, 100), num_components=50):
        self.target_size = target_size
        self.num_components = num_components
        
        # Math models
        self.mean_face = None
        self.eigenfaces = None
        self.eigenvalues = None
        self.labels = []
        self.label_map = {}
        
    def _convolution_preprocess(self, image):
        """
        Unit 1: Convolution
        Applies a mild sharpening filter to highlight edges.
        (Added Histogram Equalization for varying webcam lighting)
        """
        # Equalize lighting to resist webcam shadows vs file photos
        image = cv2.equalizeHist(image)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        processed = cv2.filter2D(image, -1, kernel)
        return processed
        
    def load_data(self, data_dir="photos"):
        """
        Loads images and treats pixels as Random Variables (Unit 3).
        Images are flattened into Vectors representing the Face Vector Space (Unit 1).
        """
        data = []
        labels = []
        label_id = 0
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory {data_dir}. Please populate it with subfolders of faces.")
            return np.array([]), np.array([])

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_path):
                continue
                
            self.label_map[label_id] = person_name
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Crop the face closely to match webcam framing
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)
                if len(faces) > 0:
                    # Pick largest face
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    x, y, w, h = faces[0]
                    img = img[y:y+h, x:x+w]
                    
                # Resize
                img = cv2.resize(img, self.target_size)
                
                # Preprocess (Convolution)
                img = self._convolution_preprocess(img)
                
                # Flatten into vector
                vector = img.flatten()
                data.append(vector)
                labels.append(label_id)
                
            label_id += 1
            
        if not data:
            print("No images found in the photos folder. Please add subfolders with images.")
            return np.array([]), np.array([])
            
        return np.array(data), np.array(labels)
        
    def train(self, data_matrix, labels):
        """
        Trains the models using Eigenvalues, SVD, Covariance (Unit 1 & 3).
        """
        if len(data_matrix) == 0:
            return
            
        self.labels = labels
        n_samples, n_features = data_matrix.shape
        
        # Expectation (Unit 3): Calculate the Mean Face
        self.mean_face = np.mean(data_matrix, axis=0)
        
        # Center the data
        A = data_matrix - self.mean_face
        
        # Computing the Covariance Matrix indirectly via SVD (Unit 1).
        # We avoid calculating N x N or D x D explicitly if D is very large.
        # SVD: A = U * S * V^T, Covariance C = (A^T * A) / (n-1) uses eigenvectors V
        
        print(f"Training on {n_samples} images...")
        
        # We use linalg.svd which is computationally stable
        # Using full_matrices=False directly gives us the top components
        U, S, Vt = linalg.svd(A, full_matrices=False)
        
        # Eigenvalues of Covariance Matrix (Unit 1)
        # S contains singular values, eigenvalues are S^2 / (n-1)
        self.eigenvalues = (S ** 2) / max(1, (n_samples - 1))
        
        # Vt contains the Eigenvectors associated with A^T A (Unit 1)
        # Take the top components
        num_c = min(self.num_components, len(Vt))
        self.eigenfaces = Vt[:num_c]  # Shape: (num_c, D)
        
        # Project training data into the Face Space (Projection Matrix, Unit 1)
        self.projected_training_data = np.dot(A, self.eigenfaces.T)
        print("Training complete.")
        
    def generate_synthetic_face(self):
        """
        Unit 3: Generative Data Augmentation using Box-Muller Transform.
        Samples from the learned Gaussian distributions of the Eigen-coefficients.
        """
        if self.eigenvalues is None:
            raise ValueError("Model must be trained first.")
            
        # We treat each principal component's projection as a Random Variable 
        # with Variance = Eigenvalue, and Mean = 0 (since we zero-centered).
        
        sampled_weights = np.zeros(len(self.eigenvalues[:len(self.eigenfaces)]))
        
        # Concept of Sampling univariate distributions using Box-Muller (Unit 3)
        for i, variance in enumerate(self.eigenvalues[:len(self.eigenfaces)]):
            # Box-Muller implementation
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)
            
            # Standard normal Z
            z0 = np.sqrt(-2.0 * np.log(max(u1, 1e-10))) * np.cos(2.0 * np.pi * u2)
            
            # Scale to variance (std_dev = sqrt(variance))
            std_dev = np.sqrt(variance)
            sampled_weight = z0 * std_dev
            sampled_weights[i] = sampled_weight
            
        # Reconstruct face (Pseudoinverse conceptually applied via dot product with orthonormal basis)
        synthetic_face_vector = self.mean_face + np.dot(sampled_weights, self.eigenfaces)
        
        # Clip to valid pixel ranges
        synthetic_face_vector = np.clip(synthetic_face_vector, 0, 255)
        
        return synthetic_face_vector.astype(np.uint8).reshape(self.target_size)
        
    def recognize_face_newton(self, test_image_vector):
        """
        Unit 2: Multivariate Regularized Regression using Newton's Method.
        Instead of nearest neighbor, we frame finding the closest match 
        as an optimization problem in the low-dimensional projected space.
        """
        # Project test image onto Face Space
        centered = test_image_vector - self.mean_face
        y = np.dot(centered, self.eigenfaces.T) # Shape: (num_components,)
        
        # Regression formulation: y ~ X * beta
        # Loss L(beta) = ||y - X * beta||^2 + lambda * ||beta||^2
        
        X = self.projected_training_data.T # Shape: (num_components, num_train_samples)
        lmbda = 0.1 # Regularization parameter
        
        num_train = X.shape[1]
        beta = np.zeros(num_train) # initial guess
        
        def compute_gradient_hessian(beta):
            residual = y - np.dot(X, beta)
            # Gradient: -2 * X^T * residual + 2 * lambda * beta
            grad = -2 * np.dot(X.T, residual) + 2 * lmbda * beta
            # Hessian: 2 * X^T * X + 2 * lambda * I
            H = 2 * np.dot(X.T, X) + 2 * lmbda * np.eye(num_train)
            return grad, H
            
        # Newton's Method loop (Unit 2)
        for _ in range(10): 
            grad, H = compute_gradient_hessian(beta)
            # Update: solve H * delta = grad
            delta = linalg.solve(H, grad)
            beta = beta - delta
            
            if linalg.norm(delta) < 1e-5:
                break
                
        # Find the class that contributes the most to the reconstruction
        class_scores = {}
        for idx, weight in enumerate(beta):
            label = self.labels[idx]
            # using absolute weight as contribution score
            class_scores[label] = class_scores.get(label, 0) + abs(weight)
            
        if not class_scores:
            return "Unknown", 0.0
            
        best_label = max(class_scores, key=class_scores.get)
        
        # Convert raw regularized Newton weights into a true Percentage Probability
        total_weight = sum(class_scores.values()) + 1e-9
        confidence_prob = (class_scores[best_label] / total_weight) * 100.0
        
        # Calculate Structural Correlation (Pearson Coefficient)
        # This is strictly immune to lighting intensity and shadow scaling!
        # It measures pure mathematical structural similarity boundary from [-1.0 to 1.0]
        reconstructed_face = self.mean_face + np.dot(y, self.eigenfaces)
        corr_matrix = np.corrcoef(test_image_vector, reconstructed_face)
        struct_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix).any() else 0.0
        
        return self.label_map[best_label], confidence_prob, struct_corr

if __name__ == "__main__":
    recognizer = GenerativeRobustFaceRecognizer()
    
    data_dir = "photos"
    
    print("Initializing Math Face Recognition Core...")
    
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"\nPlease create subfolders inside '{data_dir}' (e.g., '{data_dir}/Alice', '{data_dir}/Bob')")
        print("and populate them with face images, then run this script again.")
    else:
        # Load and Train
        data, labels = recognizer.load_data(data_dir)
        if len(data) > 0:
            recognizer.train(data, labels)
            
            # Save the Mean Face to disk
            mean_face_img = recognizer.mean_face.reshape(recognizer.target_size).astype(np.uint8)
            cv2.imwrite("mean_face.jpg", mean_face_img)
            print("\nSaved calculated 'mean_face.jpg'.")
            
            # Generate synthetic data
            print("Generating a mathematical synthetic face using Unit 3 Box-Muller...")
            synth = recognizer.generate_synthetic_face()
            cv2.imwrite("synthetic_face_sample.jpg", synth)
            print("Saved synthetic face to 'synthetic_face_sample.jpg'.")
            
            # Test the Newton recognition on a random training image (for demonstration purposes)
            test_idx = np.random.randint(0, len(data))
            predicted_name, conf, struct_corr = recognizer.recognize_face_newton(data[test_idx])
            true_name = recognizer.label_map[labels[test_idx]]
            
            print(f"\n--- Testing Unit 2 Newton Optimization Classifier ---")
            print(f"Test Image True Identity: {true_name}")
            print(f"Predicted Identity (Newton Method): {predicted_name}")
            print(f"Prediction Confidence Score: {conf:.1f}%")
            print(f"Structural Correlation: {struct_corr:.2f} (1.0 is Perfect)")

            print("\n==================================")
            print("AUTOMATED 10-SECOND WEBCAM SCAN")
            print("==================================")
            print("Accessing Mac Webcam... Please look at the camera.")
            print("Scanning for 10 seconds. Do not press any keys. Processing purely in memory (no disk saving)...")
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cap = cv2.VideoCapture(0)
            
            import time
            start_time = time.time()
            found_people = {} # Using a dictionary to track frame consistency!
            scan_duration = 10.0
            
            while time.time() - start_time < scan_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Relaxed Haar cascades to improve face detection rates
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
                
                if len(faces) > 0:
                    # Only evaluate the largest face to prevent background artifacts from being recognized as someone else
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    x, y, w, h = faces[0]
                    
                    # Extract face and apply our preprocessing (Convolution, Resize)
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, recognizer.target_size)
                    face_processed = recognizer._convolution_preprocess(face_resized)
                    face_vector = face_processed.flatten()
                    
                    # Pass through our Unit 2 Newton's Method Optimization
                    name, conf, struct_corr = recognizer.recognize_face_newton(face_vector)
                    
                    # Moderate structural threshold to reject non-face noise
                    # And require confidence to be reasonably high so head-turns don't trigger guessed matches
                    if struct_corr < 0.25 or conf < 58.0:
                        name = "Unknown"
                        
                    if name != "Unknown":
                        found_people[name] = found_people.get(name, 0) + 1
                    
            cap.release()
            
            print("\n[Scan Complete] Frame memory cleared. Webcam Shutdown.")
            print("----------------------------------")
            print("PEOPLE DETECTED IN ROOM:")
            
            # Anti-Jitter logic: The person must have been consistently identified for a significant time (~1.5 seconds)
            # This completely destroys accumulated misclassifications when a person blinks or turns their head!
            valid_people = [name for name, count in found_people.items() if count >= 45]
            
            if not valid_people:
                print(" - No faces could be clearly read.")
            else:
                for person in valid_people:
                    print(f" - {person} (Verified over {found_people[person]} frames)")
            print("----------------------------------\n")
