LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_CAMERA_MODULES := on OPENCV_INSTALL_MODULES := on
include /Users/beixinzhu/Documents/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := mixed_sample
FILE_LIST += $(wildcard $(LOCAL_PATH)/*.c)
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_SRC_FILES += $(FILE_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES := /Users/beixinzhu/Documents/OpenCV-android-sdk/sdk/native/jni/include
LOCAL_C_INCLUDES += /usr/local/include/
LOCAL_LDLIBS +=  -llog -ldl -gsl
LOCAL_STATIC_LIBRARIES := libgsl
LOCAL_LDLIBS += -L$(SYSROOT)/usr/lib -llog
include $(BUILD_SHARED_LIBRARY)






