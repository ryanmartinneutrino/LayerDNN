#ifndef __ANLAYER_H__
#define __ANLAYER_H__

#include <vector>
#include "LayerGroup.h"


class ALayerGroup : public LayerGroup
{//A layer to aggregate the output of several layers
  public:
    ALayerGroup(std::string aLayerID="noid");

    virtual ~ALayerGroup();

    virtual void initialize(bool aForce=false);
    virtual void initialize_sub_layer_info();
    virtual void initialize_pointers();
    virtual void initialize_layer_data();

    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0);


    virtual void add_layer(NLayer* aLayer){
      fLayer.push_back(aLayer);
      fnLayer=fLayer.size();
      if(fnLayer==1)fFirstLayer=aLayer;
      fLastLayer=aLayer;
      initialize_sub_layer_info();//sets fnOutput and fnNeuron
    }

    virtual void set_prev_layer_ptr(NLayer* aPrev){
      NLayer::set_prev_layer_ptr(aPrev);
      for(auto layer : fLayer)layer->set_prev_layer_ptr(aPrev);
    }

    virtual void set_prev_layer_range(counter_t aFirst=0, counter_t aLast=0){
      NLayer::set_prev_layer_range(aFirst,aLast);
      for(auto layer : fLayer)layer->set_prev_layer_range(aFirst,aLast);
    }


  protected:
    std::vector<LayerData*> fInputLayerData;//layer data of the sub layers
    std::vector<counter_t> fLayernOutput;//output of each sub layer

  private:
};

#endif // __ANLAYER_H__
