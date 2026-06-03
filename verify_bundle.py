import pickle
import sys

with open('tax_models_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

gst_model = bundle['models']['gst']
print("GST Spec:", gst_model['spec'])
print("\nARDL Coefs:")
if hasattr(gst_model['ardl']['res'], 'params'):
    print(gst_model['ardl']['res'].params.index.tolist())
print("\nARIMAX Coefs:")
if hasattr(gst_model['arimax']['res'], 'params'):
    print(gst_model['arimax']['res'].params.index.tolist())
print("\nElasticNet Features:")
print(gst_model['enet']['feature_cols'])
