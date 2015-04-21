#ifndef __ONLAYER_H__
#define __ONLAYER_H__

#include "NLayer.h"
#include "OLayerData.h"

class ONLayer : public NLayer
{ //output layer
  public:
    ONLayer();
    ONLayer(counter_t anInput, counter_t anOutput, number_t aInputBias,
            std::string aLayerID="noid");
    virtual ~ONLayer();

    //Must inherit:
    virtual void initialize(bool aForce=false);
    virtual void initialize_layer_data();
    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void update_weights(){
      fLayerData->update_weights_avgDeltaW();
    }
    //end of obligatory functions

    number_t get_cost(){
     return static_cast<OLayerData*>(fLayerData)->get_cost(fCostFunctionType);
    }
    number_t get_classification_success_rate(number_t aThreshold=0.){
      return static_cast<OLayerData*>(fLayerData)->get_classification_success_rate(aThreshold);
    }

  protected:


  private:
};

#endif // __ONLAYER_H__
