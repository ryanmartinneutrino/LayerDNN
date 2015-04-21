#ifndef __CPVLAYERGROUP_H__
#define __CPVLAYERGROUP_H__

#include "VLayerGroup.h"
#include "CNLayer.h"
#include "PNLayer.h"

class CPVLayerGroup : public VLayerGroup
{
  public:
    CPVLayerGroup();
    CPVLayerGroup(counter_t anSpan, counter_t anPool, counter_t anInput, std::string aLayerID="noid");
    virtual ~CPVLayerGroup();

    virtual CNLayer* get_convo_layer(){return fConvoLayer;}
    virtual PNLayer* get_pool_layer(){return fPoolLayer;}
  protected:
    CNLayer *fConvoLayer;
    PNLayer *fPoolLayer;
  private:
};

#endif // __CPVLAYERGROUP_H__
