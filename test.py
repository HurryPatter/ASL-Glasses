from keras.models import load_model
print(load_model("asl_model.h5").output_shape)