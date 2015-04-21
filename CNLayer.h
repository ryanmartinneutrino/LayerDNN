#ifndef __CNLAYER_H__
#define __CNLAYER_H__

#include "LNLayer.h"


class CNLayer : public LNLayer
{
  public:
    CNLayer();
    CNLayer(counter_t anSpan, counter_t anOverlap, counter_t anInput,
            std::string aLayerID="noid");
    virtual ~CNLayer();

  protected:
  private:
};

#endif // __CNLAYER_H__
