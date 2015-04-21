#ifndef __LAYERTRAINER_H__
#define __LAYERTRAINER_H__

#include "LayerDNN_Types.h"
#include "NLayer.h" //abstract base class

#include <sstream>
#include <thread>
#include <fstream>
#include <math.h>
#include <random>
#include <iostream>

class LayerTrainer
{//Class for training NLayers (including groups of NLayers which form a neural network
  public:
    LayerTrainer(NLayer *aLayer=NULL);
    virtual ~LayerTrainer();

    void train_mini_batches(TR3 &aTraining, const TR3 &aValidation, counter_t anPerBatch=0);
    void one_epoch_mini_batches(TR3 &aXY, counter_t anPerBatch=0);

    void set_layer(NLayer* aLayer){fLayer=aLayer;}
    void set_nmax_epochs(counter_t aMax){fnMaxEpoch=aMax;}
    void set_classif_threshold(number_t aT=0.5){fClassifThreshold=aT;}
    void set_reduce_LR_factor(number_t aF=1.){fReduceLRFactor=aF;}
    void set_increase_LR_factor(number_t aF=1.){fIncreaseLRFactor=aF;}
    void set_reduce_LR_term(number_t aF=0.){fReduceLRFactor=aF;}
    void set_increase_LR_term(number_t aF=0.){fIncreaseLRFactor=aF;}
    void set_classif_rate_change(number_t aR=0.){fClassifRateChange=aR;}
    void set_cost_change(number_t aR=0.){fCostChange=aR;}
    void set_target_classif_rate(number_t aR=0.99){fTargetClassifRate=aR;}

  protected:
    NLayer* fLayer;//layer to be trained
    counter_t fnMaxEpoch;//max training epochs
    number_t fClassifThreshold;//threshold to consider a particular class chosen
    number_t fReduceLRFactor;//factor to reduce learning rate
    number_t fIncreaseLRFactor;//factor to reduce learning rate
    number_t fReduceLRTerm;//term reduce learning rate
    number_t fIncreaseLRTerm;//term to reduce learning rate
    number_t fClassifRateChange;//if classification rate has changed by less than this relative amount, do something
    number_t fCostChange;
    number_t fTargetClassifRate;

  private:
};

#endif // __LAYERTRAINER_H__
