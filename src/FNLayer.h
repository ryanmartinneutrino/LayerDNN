#ifndef __FNLAYER_H__
#define __FNLAYER_H__

#include "NLayer.h"
#include "FLayerData.h"


class FNLayer : public NLayer
{
  public:
    FNLayer();
    FNLayer(counter_t anInput, counter_t anNeuron, number_t aInputBias,
            std::string aLayerID="noid");

    virtual ~FNLayer();

    virtual void initialize(bool aForce=false);
    virtual void initialize_layer_data();
    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void update_weights(){
      fLayerData->update_weights_avgDeltaW();
    }

  protected:
  private:
};

#endif // __FNLAYER_H__
