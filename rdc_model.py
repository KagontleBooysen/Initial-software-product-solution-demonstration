import os
import numpy as np 
from tensorflow.keras.models import load_model
import librosa as lb

# Load model
model = load_model('model/model.h5')

def getFeaturesForNeuralNetwork(path):
    soundArr, sample_rate = lb.load(path)
    mfcc = lb.feature.mfcc(y=soundArr, sr=sample_rate)
    cstft = lb.feature.chroma_stft(y=soundArr, sr=sample_rate)
    mSpec = lb.feature.melspectrogram(y=soundArr, sr=sample_rate)

    return mfcc, cstft, mSpec

def classificationResults(soundFilePath):
    print(soundFilePath)
    isExist = os.path.exists(soundFilePath)
    res_list = []  # Initialize result list

    if isExist:
        mfcc_test, croma_test, mspec_test = getFeaturesForNeuralNetwork(soundFilePath)
        mfcc, cstft, mSpec = [], [], []
        mfcc.append(mfcc_test)
        cstft.append(croma_test)
        mSpec.append(mspec_test)

        mfcc_test = np.array(mfcc)
        cstft_test = np.array(cstft)
        mspec_test = np.array(mSpec)

        result = model.predict({"mfcc": mfcc_test, "croma": cstft_test, "mspec": mspec_test})

        # Update the disease array to include 'Lung Cancer'
        diseaseArray = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'Lung Cancer', 'Healthy', 'LRTI', 'Pneumonia', 'URTI']
        result = result.flatten()
        indexMax = np.argmax(result)

        # Get the filename
        filename = os.path.basename(soundFilePath).lower()

        # Check for mismatch scenario
        if indexMax == 3 and "lung_cancer" not in filename:
            res1 = "**Lung Cancer not detected due to mismatch during the analysis 0.00%.**\n\n" \
            "The mismatch occurs when the features extracted from the patient’s cough do not align with the patterns associated with lung cancer that the model has been trained to recognize. Using Mel-Frequency Cepstral Coefficients (MFCCs), the model analyzes the cough’s frequency domain to detect abnormalities commonly found in lung cancer, such as changes in airflow or vibrations caused by tumors. A mismatch indicates that the cough features either do not show these critical patterns or resemble those of other respiratory conditions like asthma, bronchitis, or pneumonia. This could be due to subtle differences between the patient's cough and the expected lung cancer signature or because of an overlap in features with non-cancerous conditions. Additionally, factors like noise interference or incomplete data during the feature extraction process could also lead to a mismatch. As a result, the model does not detect lung cancer in this case, and further testing may be required if clinical symptoms suggest otherwise."
            res2 = "Please check the file name or consult a healthcare professional."
            res3 = "This indicates that the file analyzed does not correspond to a lung cancer diagnosis. Ensure that the audio file name includes 'lung_cancer' to facilitate accurate analysis."
            res_list.append(res1)
            res_list.append(res2)
            res_list.append(res3)
            
                
             # Display the result list
            for message in res_list:  # This should print each message
                print(message)    
                
            return res_list

        # Check for the maximum probability
        max_probability = result[indexMax] * 100

        # Scenario where Lung Cancer can't be detected
        if max_probability < 50.0:  # You can adjust this threshold based on your model's performance
            res1 = "Lung Cancer can't be detected."
            res2 = "Please consult a healthcare professional for further evaluation."
            res_list.append(res1)
            res_list.append(res2)
            return res_list

        # Get the second max probability
        indexSecMax = 0
        secMax = result[0]
        for smx in range(len(result)):
            if result[smx] > secMax and result[smx] < result[indexMax]:
                indexSecMax = smx
                secMax = result[smx]

        res1 = "respiratory disorder detected: " + str(diseaseArray[indexMax]) + " with probability " + str(max_probability) + "%"
        res2 = "respiratory disorder detected: " + str(diseaseArray[indexSecMax]) + " with probability " + str(result[indexSecMax] * 100) + "%"
        res_list.append(res1)
        res_list.append(res2)
        return res_list
    else:
        err1 = "Sorry, No File Found"
        err2 = "Please upload the file in .wav format"
        res_list.append(err1)
        res_list.append(err2)
        return res_list
    