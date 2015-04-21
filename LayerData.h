#ifndef __LAYERDATA_H__
#define __LAYERDATA_H__

#include "LayerDNN_Types.h"


#include <vector>
#include <iostream>

class NLayer;

//keep the choice of type for holding all the numbers flexible, as well as the way to hold arrays
//Note that the array objects, TRN, need to implement some of the stl::vector stuff.
//Inherited class should implement the initialize_ndata() method, and specific feed forward or delta calculations

class LayerData
{
  public:
    LayerData(NLayer* aLayer=NULL);
    virtual ~LayerData();
    friend std::ostream& operator<<(std::ostream& aOStream, LayerData* aLD);
    friend std::ostream& operator<<(std::ostream& aOStream, LayerData& aLD){
      aOStream<<(&aLD);
      return aOStream;
    };
    friend std::istream& operator>>(std::istream& aIStream, LayerData* aLD);
    friend std::istream& operator>>(std::istream& aIStream, LayerData& aLD){
      aIStream>>(&aLD);
      return aIStream;
    }
    virtual void print(counter_t aDataIndex=0);

    virtual void initialize_weights(bool aRandomize=true, number_t aFixedW=0.);
    virtual void initialize_ndata(const counter_t anData=1, bool aForce=false);

    virtual void copy_data_to_output(const TR2 &aX, counter_t aStart=0, counter_t anData=0);
    virtual void copy_data_to_output(const std::vector<LayerData*> &aInputLayerData);
    virtual void feed_forward(const TR2 &aX, counter_t aStart=0, counter_t anData=0, ActivationType aType=kLogisticActivation);
    virtual void feed_forward(const LayerData &aPrevLayer, ActivationType aType=kLogisticActivation);
    virtual void feed_forward_pool(LayerData &aPrevLayer);//not const, because it tells the previous layer which was its max output

    virtual void just_delta_from_next_layer(const LayerData &aNextLayer);
    virtual void delta_from_next_layer(const LayerData &aPrevLayer, const LayerData &aNextLayer,
                               ActivationType aType=kLogisticActivation);
    virtual void update_weights_avgDeltaW();

    virtual NLayer* get_layer(){return fLayer;};
    virtual void set_learning_rate_gain(number_t al){
        if(fLearningRateGain.size()<1)return;
        for(counter_t i=0;i<fLearningRateGain.size();i++){
          for(counter_t j=0;j<fLearningRateGain[i].size();j++)fLearningRateGain[i][j]=al;
        }
    }

    protected:
      NLayer* fLayer;
      LayerData* fInputLayerData;//used when passed data to forward_pass instead of a previous layer

    public://These need to be public so that derived classes can access each other's data

      //Weight matrix and gradient, these are fnNeurons x fnInput
      TR2 fW;//weights don't change for each data sets
      TR2 fAvgDeltaW;//average gradient, this is what actually gets applied during weight update
      TR2 fAvgDeltaWPrev;//previous average gradient, for momentum,
      TR2 fLearningRateGain;//one learning rate per weight

      //Stuff that gets updates for each data set, dimensions are fnData x fnNeuron
      TR2 fOutput;//outputs from this layer (Y)
      TR2 fZ;//weighted inputs (Z)
      TR2 fDelta;
      TR3 fDeltaW;//Delta W (one per data set, per weight)

      TR2 fOutputFactor;//used to scale output when next layer in a pooling layer

};

#endif // __LAYERDATA_HH__
