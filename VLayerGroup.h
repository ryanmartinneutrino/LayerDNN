#ifndef __VLAYERGROUP_H__
#define __VLAYERGROUP_H__

#include "LayerGroup.h"
#include <iostream>


class VLayerGroup : public LayerGroup
{
  //Vertical stack of layers, where the output of one is the output of the next
  public:
    VLayerGroup(std::string aLayerID="noid");
    virtual ~VLayerGroup();

    virtual void initialize(bool aForce=false);
    virtual void initialize_pointers();
    virtual void initialize_layer_data();

    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0);
    virtual void copy_targets(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0);

    virtual number_t get_cost(){return fLastLayer->get_cost();}
    virtual number_t get_classification_success_rate(number_t aThreshold=0.){
      return fLastLayer->get_classification_success_rate(aThreshold);};

    void add_layer(NLayer* aLayer){
      fLayer.push_back(aLayer);
      fnLayer=fLayer.size();
      if(fnLayer==1)fFirstLayer=aLayer;
      fLastLayer=aLayer;
      fnOutput=fLastLayer->get_noutput();
      fnInput=fFirstLayer->get_ninput();
    }

    //For a vertical group, the first layer is assumed to see the layer that is previous to the group
    //and the last layer should see the next layer from the group:
    virtual void set_next_layer_ptr(NLayer* aNext){
      fNextLayer=aNext;
      if(fLastLayer!=NULL)fLastLayer->set_next_layer_ptr(aNext);
      else ERROR("last layer pointer is null");
    }
    virtual void set_prev_layer_ptr(NLayer* aPrev){
      fPrevLayer=aPrev;
      if(fFirstLayer!=NULL)fFirstLayer->set_prev_layer_ptr(aPrev);
      else ERROR("first layer pointer is null");
    }
    virtual void set_next_layer_range(counter_t aFirst=0, counter_t aLast=0){
      NLayer::set_next_layer_range(aFirst, aLast);
      if(fLastLayer!=NULL)fLastLayer->set_next_layer_range(aFirst,aLast);
      else ERROR("last layer pointer is null");
    }
    virtual void set_prev_layer_range(counter_t aFirst=0, counter_t aLast=0){
      NLayer::set_prev_layer_range(aFirst,aLast);
      if(fFirstLayer!=NULL)fFirstLayer->set_prev_layer_range(aFirst,aLast);
      else ERROR("first layer pointer is null");
    }
  protected:

  private:
};

#endif // __VLAYERGROUP_H__
