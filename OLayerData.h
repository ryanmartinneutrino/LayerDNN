#ifndef __OLAYERDATA_H__
#define __OLAYERDATA_H__

#include "LayerData.h"

class ONLayer;

class OLayerData : public LayerData
{//same as LayerData, but with the ability to hold target values.
  public:
    OLayerData(ONLayer* aLayer=NULL);
    virtual ~OLayerData();

    virtual void initialize_ndata(const counter_t anData=1, bool aForce=false);

    void copy_data_to_target(const TR2 &aY, counter_t aStart=0, counter_t anData=0);

    virtual void delta_from_target(const LayerData &aPrevLayer,
                                    ActivationType aType=kLogisticActivation,
                                    CostFunctionType aCostType=kQuadraticCost);

    number_t get_cost(CostFunctionType aCostType=kQuadraticCost);
    number_t get_classification_success_rate(number_t aThreshold=0.);

  public:
    TR2 fTarget;//hold target values

  private:
};

#endif // __OLAYERDATA_H__
