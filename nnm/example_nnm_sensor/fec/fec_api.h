/*
 *******************************************************************************
 *  Copyright (c) 2010-2021 VATICS Inc. All rights reserved.
 *
 *  +-----------------------------------------------------------------+
 *  | THIS SOFTWARE IS FURNISHED UNDER A LICENSE AND MAY ONLY BE USED |
 *  | AND COPIED IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF SUCH  |
 *  | A LICENSE AND WITH THE INCLUSION OF THE THIS COPY RIGHT NOTICE. |
 *  | THIS SOFTWARE OR ANY OTHER COPIES OF THIS SOFTWARE MAY NOT BE   |
 *  | PROVIDED OR OTHERWISE MADE AVAILABLE TO ANY OTHER PERSON. THE   |
 *  | OWNERSHIP AND TITLE OF THIS SOFTWARE IS NOT TRANSFERRED.        |
 *  |                                                                 |
 *  | THE INFORMATION IN THIS SOFTWARE IS SUBJECT TO CHANGE WITHOUT   |
 *  | ANY PRIOR NOTICE AND SHOULD NOT BE CONSTRUED AS A COMMITMENT BY |
 *  | VATICS INC.                                                     |
 *  +-----------------------------------------------------------------+
 *
 *******************************************************************************
 */

#ifndef FEC_API_H
#define FEC_API_H

#include <sys/stat.h>
#include <video_source.h>
#include <gyro_daemon.h>
#include <iniparser/iniparser.h>
//#include "kp_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct FEC_INIT
 * @brief  The structure of fec initialize parameters
 */
typedef struct
{
    char* pszSensorConfigPath;          //! Path of sensor config file
    char* pszAutoSceneConfigPath;       //! Path of autoscene_config_file config file
    char* pszFusionConfigPath;          //! Path of sensor config file for fusion mode
    char* pszFecCalibratePath;          //! Path of fec calibrate config file
    char* pszFecConfPath;               //! Path of fec config file
    unsigned int dwFecMode;             //! Fec Mode
    unsigned int dwFecAppType;          //! Fec App Type
} FEC_INIT_T;

/**
 * @struct   FEC_MODE
 * @brief    The structure for FEC MODE.
 */
typedef enum
{
    FEC_MODE_1O = 0,
    FEC_MODE_1R,
    FEC_MODE_180_A,
    FEC_MODE_180_O,
    FEC_MODE_PT
} FEC_MODE;

/**
 * @struct	Extra_PT
 * @brief	The structure for Extra_PT structure
 */
typedef struct
{
    unsigned int bExtraPtEn;
    unsigned int dwX[4];
    unsigned int dwY[4];
} Extra_PT;

/**
 * @struct	EIS_T
 * @brief	The structure for EIS structure
 */
typedef struct
{
    char* pszCurveNodesPath;
    float fGyroDataGain;
    unsigned int dwGridSection;
    unsigned int dwMaxGridSection;
    float fCropRatio;
    unsigned int dwImageType;
    unsigned int dwProcessMode;
    unsigned int dwCoordinateTransform[3];
    long long sqwTimeOffset;
    unsigned int bImageRotate180;
    int sdwReadoutTimeOffset;              //! Offset of VIC readout time
    float fReadoutTimeRatio;               //! Ratio of fusion readout time. Default: 1.0
    unsigned int bForceOriRs;
    VMF_VSRC_GYRO_CONFIG_T tGyroConfig;
}EIS_T;

typedef struct
{
    //! FEC default parameters
    unsigned int lens_type;
    unsigned int app_type;

    float orig_focal;
    float orig_zoom;
    float orig_src_radius;
    float orig_dst_radius;
    float orig_convex_ratio_v;
    float orig_convex_ratio_h;
    float orig_depth_ratio;
    Extra_PT orig_extra_pt;

    //! p180 all direction mode
    float p180a_tilt;
    float p180a_zoom;
    float p180a_focal;
    float p180a_spin;
    Extra_PT p180a_extra_pt;

    //! p180 one direction mode
    float p180o_tilt;
    float p180o_zoom;
    float p180o_focal;
    float p180o_spin;
    Extra_PT p180o_extra_pt;

    //! p180 offset
    float p180_dst_offset_x;
    float p180_dst_offset_y;
    float p180_dst_xy_ratio; //! range:  0.1 ~ 2

    //! 1r mode
    float m1r_tilt;
    float m1r_zoom;
    float m1r_focal;
    float m1r_pan_wall;
    float m1r_tilt_wall;
    float m1r_dst_rotate;
    Extra_PT m1r_extra_pt;

    //! PT mode
    char* pszDatFile;
    char* pszComplexDatFile;
    char* pszMrfDatFile;

    //! EIS 
    EIS_T tEis;
} FECDefValue;

int setup_fec_mode(VMF_VSRC_HANDLE_T*, VMF_LAYOUT_T*, FEC_MODE, unsigned int);
int loadFECConfig(FEC_INIT_T *, VMF_VSRC_FRONTEND_CONFIG_T *);
int loadCalibrateConfig(FEC_INIT_T *, VMF_VSRC_FRONTEND_CONFIG_T *);
void free_fec_def_str(void);
size_t get_lens_curve_nodes(const char* filename, float* nodes_x, float* nodes_y);
#ifdef __cplusplus
}
#endif

#endif
