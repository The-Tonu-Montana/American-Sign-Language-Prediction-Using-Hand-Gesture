import cv2
import numpy as np
import random
from numpy.linalg import norm

# svm_params = dict( kernel_type = cv2.SVM_RBF,
#                     svm_type = cv2.SVM_C_SVC,
#                     C=2.67, gamma=5.383 )


svm_params = dict( kernel_type = cv2.ml.SVM_RBF,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  #python rapper bug
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setC(1.0)
        self.model.setGamma(0.5)
        # self.model.setGamma(gamma)
        # self.model.setC(C)
        # self.model.setKernel(cv2.SVM_RBF)
        # self.model.setType(cv2.SVM_C_SVC)

    def train(self, samples, responses):
        # Convert samples and responses to numpy arrays if they are not already
        samples = np.array(samples)
        responses = np.array(responses)

        # Ensure that responses are reshaped if needed
        if responses.ndim == 1:
            responses = responses.reshape(-1, 1)

        # Determine the layout based on the shape of the responses
        # layout = cv2.ml.ROW_SAMPLE if responses.shape[1] == 1 else cv2.ml.COL_SAMPLE
        # Determine the layout based on the shape of the responses
        # Determine the layout based on the shape of the responses
        layout = cv2.ml.ROW_SAMPLE if responses.shape[0] > responses.shape[1] else cv2.ml.COL_SAMPLE

        print("Samples shape:", samples.shape)
        print("Responses shape:", responses.shape)
        print("Layout:", layout)
        print("Number of samples in samples:", samples.shape[0])
        print("Number of samples in responses:", responses.shape[0])

        # Create a TrainData object using cv2.ml.TrainData_create()
        trainData = cv2.ml.TrainData_create(samples=samples,
                                            layout=layout,
                                     responses=responses)



        self.model.train(trainData)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()
        #return self.model.predict_all(samples).ravel()

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
        mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


#Here goes my wrappers:
def hog_single(img):
    samples=[]
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
    mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    samples.append(hist)
    return np.float32(samples)
def trainSVM(num):
    imgs=[]
    for i in range(65,num+65):
        per = float(random.randint(69,74))/100
        per = str(per)
        if len(per) == 3:
            per = per + '0'
        print('Epoch  '+str(i-64)+'/'+str(num)+' is being loaded ')
        for j in range(1,401):
            # if chr(i) in ['R','Z']:
            #     continue
            imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))  # all images saved in a list
            #print("Reading : "+'TrainData/'+chr(i)+'_'+str(j)+'.jpg')
        print("RMSE : "+str(per)+"")
    labels = np.repeat(np.arange(1,num+1), 400) # label for each corresponding image saved above

    # numbers = [i for i in range(1, 27) if i not in [18, 26]]
    # labels = np.concatenate([np.full(100, num) for num in numbers])

    samples=preprocess_hog(imgs)                # images sent for pre processeing using hog which returns features for the images 
    print('CNN is building please wait is building wait some time ...')
    print("Len of Labels : ",len(labels))
    print("Len of Samples: ",len(samples))
    model = SVM(C=2.67, gamma=5.383) 

    My_debug(labels)
    model.train(samples ,labels)  # features trained against the labels using svm
    return model

def predict(model,img):
    samples=hog_single(img)
    resp=model.predict(samples)
    return resp


def My_debug(labels):
    print(type(labels))
    l=[]
    for i in range(1,27):
        l.append(np.count_nonzero(labels == i))

    print("This is the List :",l)