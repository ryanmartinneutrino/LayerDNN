#ifndef __LNLAYER_H__
#define __LNLAYER_H__

#include "NLayer.h"


class LNLayer : public NLayer
{
  //Local field receptor. Each neuron only looks at a fraction of the input (the Span), the span
  //of different neurons can overlap
  public:
    LNLayer();
    LNLayer(counter_t anSpan, counter_t anOverlap, counter_t anInput, number_t aInputBias=0.,
            std::string aLayerID="noid");
    virtual ~LNLayer();

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

#endif // __LNLAYER_H__
