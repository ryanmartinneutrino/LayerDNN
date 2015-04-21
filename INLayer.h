#ifndef __INLAYER_H__
#define __INLAYER_H__

#include "NLayer.h"
//#include "ILayerData.h"

#include <iostream>

class INLayer : public NLayer
{
  //Input layer that just copies data to its output on a forward pass
  public:
    INLayer();
    INLayer(counter_t anInput, std::string aLayerID="noid");
    virtual ~INLayer();

    virtual void initialize(bool aForce=false);
    virtual void initialize_layer_data();
    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0){}//do nothing
    virtual void update_weights(){}

  protected:
  private:
};

#endif // __INLAYER_H__
