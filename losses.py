# importing required libraries
from tensorflow.keras import backend as K
import tensorflow as tf

class Loss:
    def __init__(self):
        self.smooth = 1.
    
    def diceCoef(self, y_true, y_pred):   
        y_true_f = K.flatten(y_true)    
        y_pred_f = K.flatten(y_pred)    
        intersection = K.sum(y_true_f * y_pred_f)    
        
        return (2. * intersection + self.smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + self.smooth)

    def truePositive(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        return (K.sum(y_pos * y_pred_pos) + self.smooth)/ (K.sum(y_pos) + self.smooth)  
        
    def trueNegative(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos 
        return (K.sum(y_neg * y_pred_neg) + self.smooth) / (K.sum(y_neg) + self.smooth)
        
    def falsePositive(self, trueNegative):
        return 1 - trueNegative
        
    def falseNegative(self, truePositive):
        return 1 - truePositive
        
    def IoULoss(self, targets, inputs, smooth=1e-6):
    
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        intersection = K.sum(targets * inputs)
        total = K.sum(targets) + K.sum(inputs)
        union = total - intersection
        
        IoU = (intersection + smooth) / (union + smooth)
        return IoU
        
    def PanopticLoss(self, y_true, y_pred):
        truePositive = self.truePositive(y_true, y_pred)
        trueNegative = self.trueNegative(y_true, y_pred)

        falsePositive = self.falsePositive(trueNegative)
        falseNegative = self.falseNegative(truePositive)

        iou = self.IoULoss(y_true, y_pred, smooth=0.25)
        denom = truePositive + 0.5 * falsePositive + 0.5 * falseNegative
        return 1. - ((iou/truePositive) * (truePositive/denom))
        
    def diceCoefLoss(self, y_true, y_pred):
        return 1. - self.diceCoef(y_true, y_pred)
        
    def slicesPanopticLoss(self, y_true, y_pred):
        losses = []
        for i in range(6):
            loss = self.PanopticLoss(y_true[:,:,:,i], y_pred[:,:,:,i])
            losses.append(loss)
        return tf.experimental.numpy.mean(losses)
    
    def slicesPanopticScore(self, y_true, y_pred):
        return 1 - self.slicesPanopticLoss(y_true, y_pred)