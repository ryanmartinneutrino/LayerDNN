#ifndef __NLAYER_H__
#define __NLAYER_H__

#include "LayerDNN_Types.h"
#include "LayerData.h"
#include <string>
#include <iostream>
#include <stdlib.h>


class NLayer
{
//This is a general class to manage a layer in a deep neural network. All the data for the layer
//all held in a LayerData class (to allow specific optimization of the calculations)
//This is an abstract class
//It can also be used to hold an array of layers
  public:
    NLayer(std::string aLayerID="noid");
    virtual ~NLayer();

    //for streaming NLayers
    friend std::ostream& operator<<(std::ostream& aOStream, NLayer* aL);
    friend std::ostream& operator<<(std::ostream& aOStream, NLayer& aL){
      aOStream<<(&aL);
      return aOStream;
    };
    friend std::istream& operator>>(std::istream& aIStream, NLayer* aL);
    friend std::istream& operator>>(std::istream& aIStream, NLayer& aL){
      aIStream>>(&aL);
      return aIStream;
    }

    virtual void print(counter_t aDataIndex=0);


    //!!Methods that inherited classes need to implement:
    //initialize data members to the specific values for inherited classes (e.g. input layer has no neurons)
    virtual void initialize(bool aForce=false)=0;
    //initialize the layer data to be the right size:
    virtual void initialize_layer_data()=0;
    //each inherited layer must define the 3 following methods for backprop training:
    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0)=0;
    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0)=0;
    virtual void update_weights()=0;

    //for LayerGroup derived classes and aggregating layer
    virtual void copy_targets(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0){};
    virtual void add_layer(NLayer* aLayer){
      fLayer.push_back(aLayer);
      fnLayer=fLayer.size();
    }

    //for output layer derived classes:
    virtual number_t get_cost(){return -1.0;}
    virtual number_t get_classification_success_rate(number_t aThreshold=0.){return -1.0;}

    //Note that some getters and setters are overwritten in derived classes (in particular those
    //that set prev and next layer pointers and ranges)
    //Getters
    virtual LayerData* get_layer_data_ptr(){return fLayerData;}
    virtual counter_t get_ninput(){return fnInput;}
    virtual counter_t get_noutput(){return fnOutput;}
    virtual counter_t get_nneuron(){return fnNeuron;}
    virtual counter_t get_nweight(){return fnWeight;}
    virtual counter_t get_ndata(){return fnData;}
    virtual counter_t get_nlayer(){return fnLayer;}
    virtual std::string get_layer_id(){return fLayerID;}
    virtual NLayer* get_next_layer_ptr(){return fNextLayer;}
    virtual NLayer* get_prev_layer_ptr(){return fPrevLayer;}
    virtual NLayer* get_last_layer(counter_t aIndex=0){
      return (fnLayer==0 ? this : fLayer[fnLayer-1]);
    }
    virtual std::string get_new_layer_id();
    virtual counter_t get_next_layer_first(){return fNextLayerRange.first;}
    virtual counter_t get_next_layer_last(){return fNextLayerRange.last;}
    virtual counter_t get_prev_layer_first(){return fPrevLayerRange.first;}
    virtual counter_t get_prev_layer_last(){return fPrevLayerRange.last;}
    virtual counter_t get_convo_delta(){return fConvoPars.span-fConvoPars.overlap;}
    virtual counter_t get_convo_overlap(){return fConvoPars.overlap;}
    virtual counter_t get_convo_span(){return fConvoPars.span;}
    virtual number_t get_input_bias(){return fInputBias;}
    virtual ActivationType get_activation_type(){return fActivationType;}
    virtual LayerType get_layer_type(){return fLayerType;}
    virtual CostFunctionType get_cost_function_type(){return fCostFunctionType;}
    virtual number_t get_global_learning_rate(){return fGlobalLearningRate;}
    virtual number_t get_momentum_alpha(){return fMomentumAlpha;}
    virtual number_t get_L2_regularization(){return fL2Reg;}
    virtual TR2 get_all_output_data(){return fLayerData->fOutput;}
    virtual std::string get_layer_type_str(LayerType aType=kUninitializedLayer);
    virtual std::string get_activation_type_str(){
      switch(fActivationType){
        case kUninitializedActivation:
          return "Uninitialized";
        case kLogisticActivation:
          return "Logistic";
        case kReLUActivation:
          return "Rectified Linear";
        case kSoftMaxActivation:
          return "SoftMax";
        default:
          return "Unknown";
      }
    }
    virtual std::string get_cost_function_type_str(){
      switch(fCostFunctionType){
        case kUninitializedCost:
          return "Uninitialized";
        case kQuadraticCost:
          return "Quadratic";
        case kXCorrelCost:
          return "Cross-correlation";
        default:
          return "Unknown";
      }
    }
    //Setters
    virtual void set_layer_ID(std::string aID){fLayerID=aID;}
    virtual void set_layer_type(LayerType aType){fLayerType=aType;}
    virtual void set_activation_type(ActivationType aType){fActivationType=aType;}
    virtual void set_cost_function_type(CostFunctionType aType){fCostFunctionType=aType;}
    virtual void set_next_layer_ptr(NLayer* aNext){fNextLayer=aNext;}
    virtual void set_prev_layer_ptr(NLayer* aPrev){fPrevLayer=aPrev;}
    virtual void set_ndata(counter_t anData=0){fnData=anData;}
    virtual void set_next_layer_range(counter_t aFirst=0, counter_t aLast=0){
      fNextLayerRange.first=aFirst;
      fNextLayerRange.last=aLast;}
    virtual void set_prev_layer_range(counter_t aFirst=0, counter_t aLast=0){
      fPrevLayerRange.first=aFirst;
      fPrevLayerRange.last=aLast;}
    virtual void set_global_learning_rate(number_t aRate){fGlobalLearningRate=aRate;}
    virtual void multiply_global_learning_rate(number_t aFact=1.0){fGlobalLearningRate*=aFact;}
    virtual void add_to_global_learning_rate(number_t a=0.0){fGlobalLearningRate+=a;}
    virtual void set_momentum_alpha(number_t aAlpha){fMomentumAlpha=aAlpha;}
    virtual void set_L2_regularization(number_t aL2){fL2Reg=aL2;}

  protected:
    std::string fLayerID;//Unique layer ID, could be used to get layers in memory
    bool fInitialized=false;

    counter_t fnNeuron;
    counter_t fnInput;
    counter_t fnOutput;
    counter_t fnWeight;//number of weights per neuron (this could be different than fnInput if
                       //there is an input bias or if the neurons in this layer do not span
                       //the full range of outputs of a previous layer (e.g. a convolutional layer)

    IndexRange fPrevLayerRange;//range of indices of neurons in the previous layer to use as inputs
    IndexRange fNextLayerRange;//range indices for neurons in the next layer that receive this layer as input
                               //which is needed for backprop. Typically, this should be set to 0 and the index
                               //of the last neuron in the next layer (nNeurons-1 of the next layer)
    ConvoPars fConvoPars;//used if this is a local receptive field

    LayerType fLayerType;
    number_t fInputBias;//value of bias for neurons in this layer (=0 -> no bias!)
    counter_t fnData;
    ActivationType fActivationType;
    CostFunctionType fCostFunctionType;//XCorrel is default

    LayerData *fLayerData;//data for the layer (weights, outputs, etc)l

    NLayer *fPrevLayer;
    NLayer *fNextLayer;

    number_t fMomentumAlpha;//alpha constant for momentum
    number_t fGlobalLearningRate;
    number_t fL2Reg;//lambda parameter in L2 regularization

    //These are for Layer groups to inherit, derive a class from LayerGroup to use these!!!
    //They are located here so that the derived classes can use the base class stream operator
    std::vector<NLayer*> fLayer;
    counter_t fnLayer;

    counter_t fInstanceCount;//count instances of a class
};


#endif // __NLAYER_H__
