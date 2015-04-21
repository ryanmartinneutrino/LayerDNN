#ifndef __PNLAYER_H__
#define __PNLAYER_H__

#include "NLayer.h"


class PNLayer : public NLayer
{
  public:
    PNLayer();
    PNLayer(counter_t anPool, counter_t anInput, std::string aLayerID="noid");
    virtual ~PNLayer();

    virtual void initialize(bool aForce=false);
    virtual void initialize_layer_data();
    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void update_weights(){};

  protected:

  private:
};

#endif // __PNLAYER_H__
