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
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <unistd.h>

#include <video_source.h>
#include <fec_layout.h>
#include <video_buf.h>
#include <video_encoder.h>
#include <iniparser/iniparser.h>
#include "fec_api.h"

FECDefValue gFecDefValue;
//static EIS_T g_tEis;

static void setup_fec_layout(VMF_FEC_LYT_CONFIG_T* ptFecLayout, VMF_LAYOUT_T* ptLayout, VMF_FEC_METHOD eFecMethod)
{
    ptFecLayout->dwOutputId = 0;
    ptFecLayout->tLayout.dwCanvasWidth = ptLayout->dwCanvasWidth;
    ptFecLayout->tLayout.dwCanvasHeight = ptLayout->dwCanvasHeight;
    ptFecLayout->tLayout.dwVideoWidth = ptLayout->dwVideoWidth;
    ptFecLayout->tLayout.dwVideoHeight = ptLayout->dwVideoHeight;
    ptFecLayout->tLayout.dwVideoPosX = ptLayout->dwVideoPosX;
    ptFecLayout->tLayout.dwVideoPosY = ptLayout->dwVideoPosY;
    ptFecLayout->dwClearBackColor = 0; //! full FEC layout dont need to setup back color
    ptFecLayout->eGridSize = VMF_FEC_GRID_8X8;
    ptFecLayout->eLayoutMethod = eFecMethod;
}

static int setup_fec_1o(VMF_VSRC_HANDLE_T* ptVsrcHandle, VMF_LAYOUT_T* ptLayout, unsigned int dwEisEnable)
{
    VMF_FEC_ORIG_CONFIG_T  tFecOrgConfig;
    VMF_FEC_CELL_CONFIG_T tFecCellConfig;
    VMF_FEC_LYT_CONFIG_T  tFecLytConfig;

    memset(&tFecOrgConfig, 0, sizeof(VMF_FEC_ORIG_CONFIG_T) );
    memset(&tFecCellConfig, 0, sizeof(VMF_FEC_CELL_CONFIG_T));
    memset(&tFecLytConfig, 0, sizeof(VMF_FEC_LYT_CONFIG_T));

    setup_fec_layout(&tFecLytConfig, ptLayout, VMF_FEC_METHOD_GTR);

    tFecOrgConfig.fZoom = gFecDefValue.orig_zoom;
    tFecOrgConfig.dwSrcRadius = gFecDefValue.orig_src_radius;
    tFecOrgConfig.dwOutRadius = gFecDefValue.orig_dst_radius;
    tFecOrgConfig.fVerCvxRatio = gFecDefValue.orig_convex_ratio_v;
    tFecOrgConfig.fHorCvxRatio = gFecDefValue.orig_convex_ratio_h;
    tFecOrgConfig.fDepthRatio = gFecDefValue.orig_depth_ratio;

    tFecOrgConfig.eAppType = (VMF_FEC_APP_TYPE)gFecDefValue.app_type;
    memcpy(&tFecOrgConfig.tExtraPt, &gFecDefValue.orig_extra_pt, sizeof(Extra_PT));

    tFecCellConfig.eFecMode = VMF_FEC_COEF_MODE_ORIG;
    tFecCellConfig.bEisMode = dwEisEnable;
    tFecCellConfig.pFecConfig = &tFecOrgConfig;

    VMF_FEC_LYT_Single(ptVsrcHandle, &tFecLytConfig, &tFecCellConfig);

    return 0;
}

static int setup_fec_1r(VMF_VSRC_HANDLE_T* ptVsrcHandle, VMF_LAYOUT_T* ptLayout, unsigned int dwEisEnable)
{
    VMF_FEC_PTZ_CONFIG_T  tFecPtzConfig;
    VMF_FEC_CELL_CONFIG_T tFecCellConfig;
    VMF_FEC_LYT_CONFIG_T  tFecLytConfig;

    memset(&tFecPtzConfig, 0, sizeof(VMF_FEC_PTZ_CONFIG_T) );
    memset(&tFecCellConfig, 0, sizeof(VMF_FEC_CELL_CONFIG_T));
    memset(&tFecLytConfig, 0, sizeof(VMF_FEC_LYT_CONFIG_T));

    if (VMF_FEC_APP_WALL == tFecPtzConfig.eAppType) {
        tFecPtzConfig.fPan  = gFecDefValue.m1r_pan_wall;
        tFecPtzConfig.fTilt = gFecDefValue.m1r_tilt_wall;
    } else {
        tFecPtzConfig.fTilt = gFecDefValue.m1r_tilt;
    }
    tFecPtzConfig.fZoom = gFecDefValue.m1r_zoom;
    tFecPtzConfig.fFocalLength = gFecDefValue.m1r_focal;
    tFecPtzConfig.eAppType = (VMF_FEC_APP_TYPE)gFecDefValue.app_type;
    tFecPtzConfig.eLensType = (VMF_FEC_LENS_TYPE)gFecDefValue.lens_type;
    tFecPtzConfig.fDstRotate = gFecDefValue.m1r_dst_rotate;
    memcpy(&tFecPtzConfig.tExtraPt, &gFecDefValue.m1r_extra_pt, sizeof(Extra_PT));

    tFecCellConfig.eFecMode = VMF_FEC_COEF_MODE_PTZ;
    tFecCellConfig.bEisMode = dwEisEnable;
    tFecCellConfig.pFecConfig = &tFecPtzConfig;

    setup_fec_layout(&tFecLytConfig, ptLayout, VMF_FEC_METHOD_GTR);

    VMF_FEC_LYT_Single(ptVsrcHandle, &tFecLytConfig, &tFecCellConfig);

    return 0;
}

static int setup_fec_p180a(VMF_VSRC_HANDLE_T* ptVsrcHandle, VMF_LAYOUT_T* ptLayout)
{
    VMF_FEC_P180_CONFIG_T tFecP180Config;
    VMF_FEC_CELL_CONFIG_T tFecCellConfig;
    VMF_FEC_LYT_CONFIG_T  tFecLytConfig;

    memset(&tFecP180Config, 0, sizeof(VMF_FEC_P180_CONFIG_T));
    memset(&tFecCellConfig, 0, sizeof(VMF_FEC_CELL_CONFIG_T));
    memset(&tFecLytConfig, 0, sizeof(VMF_FEC_LYT_CONFIG_T));

    //! setup P180a cell config parameters

    tFecP180Config.fTilt = gFecDefValue.p180a_tilt;
    tFecP180Config.fZoom = gFecDefValue.p180a_zoom;
    tFecP180Config.fFocalLength = gFecDefValue.p180a_focal;
    tFecP180Config.eModeType = VMF_FEC_MODE_PANO_180_ALL_DIRECTION;
    tFecP180Config.fDstOffsetX = gFecDefValue.p180_dst_offset_x;
    tFecP180Config.fDstOffsetY = gFecDefValue.p180_dst_offset_y;
    tFecP180Config.fDstXYRatio = gFecDefValue.p180_dst_xy_ratio;
    tFecP180Config.eLensType = (VMF_FEC_LENS_TYPE)gFecDefValue.lens_type;
    tFecP180Config.fSpin = gFecDefValue.p180a_spin;
    memcpy(&tFecP180Config.tExtraPt, &gFecDefValue.p180a_extra_pt, sizeof(Extra_PT));
    tFecCellConfig.eFecMode = VMF_FEC_COEF_MODE_P180;
    tFecCellConfig.pFecConfig = &tFecP180Config;
    
    setup_fec_layout(&tFecLytConfig, ptLayout, VMF_FEC_METHOD_GTR);

    //! setup layout
    VMF_FEC_LYT_Single(ptVsrcHandle, &tFecLytConfig, &tFecCellConfig);

    return 0;
}

static int setup_fec_p180o(VMF_VSRC_HANDLE_T* ptVsrcHandle, VMF_LAYOUT_T* ptLayout, unsigned int dwEisEnable)
{
    VMF_FEC_P180_CONFIG_T tFecP180Config;
    VMF_FEC_CELL_CONFIG_T tFecCellConfig;
    VMF_FEC_LYT_CONFIG_T  tFecLytConfig;

    memset(&tFecP180Config, 0, sizeof(VMF_FEC_P180_CONFIG_T));
    memset(&tFecCellConfig, 0, sizeof(VMF_FEC_CELL_CONFIG_T));
    memset(&tFecLytConfig, 0, sizeof(VMF_FEC_LYT_CONFIG_T));

    //! setup P180o cell config parameters
    tFecP180Config.fTilt = gFecDefValue.p180o_tilt;
    tFecP180Config.fZoom = gFecDefValue.p180o_zoom;
    tFecP180Config.fFocalLength = gFecDefValue.p180o_focal;
    tFecP180Config.eModeType = VMF_FEC_MODE_PANO_180_ONE_DIRECTION;
    tFecP180Config.fDstOffsetX = gFecDefValue.p180_dst_offset_x;
    tFecP180Config.fDstOffsetY = gFecDefValue.p180_dst_offset_y;
    tFecP180Config.fDstXYRatio = gFecDefValue.p180_dst_xy_ratio;
    tFecP180Config.eLensType = (VMF_FEC_LENS_TYPE)gFecDefValue.lens_type;
    tFecP180Config.fSpin = gFecDefValue.p180o_spin;
    memcpy(&tFecP180Config.tExtraPt, &gFecDefValue.p180o_extra_pt, sizeof(Extra_PT));
    tFecCellConfig.eFecMode = VMF_FEC_COEF_MODE_P180;
    tFecCellConfig.bEisMode = dwEisEnable;
    tFecCellConfig.pFecConfig = &tFecP180Config;

    setup_fec_layout(&tFecLytConfig, ptLayout, VMF_FEC_METHOD_GTR);

    //! setup layout
    VMF_FEC_LYT_Single(ptVsrcHandle, &tFecLytConfig, &tFecCellConfig);

    return 0;
}

static int setup_fec_pt_mode(VMF_VSRC_HANDLE_T* ptVsrcHandle, VMF_LAYOUT_T* ptLayout)
{
    VMF_FEC_PT_CONFIG_T   tFecPtConfig;
    VMF_FEC_CELL_CONFIG_T tFecCellConfig;
    VMF_FEC_LYT_CONFIG_T  tFecLytConfig;
    FILE *fpDat = NULL;
    struct stat st;

    memset(&tFecPtConfig, 0, sizeof(VMF_FEC_PT_CONFIG_T));
    memset(&tFecCellConfig, 0, sizeof(VMF_FEC_CELL_CONFIG_T));
    memset(&tFecLytConfig, 0, sizeof(VMF_FEC_LYT_CONFIG_T));

    fpDat = fopen(gFecDefValue.pszDatFile, "r");
    if (fpDat) {
        if (fstat(fileno(fpDat), &st) == 0) {
            tFecPtConfig.dwCoeffDataSize = st.st_size;
            tFecPtConfig.pCoeffData = calloc(1, st.st_size);
            if (tFecPtConfig.pCoeffData) {
                if (fread(tFecPtConfig.pCoeffData, tFecPtConfig.dwCoeffDataSize, 1, fpDat) != 1) {
                    printf("Read %s failed\n", gFecDefValue.pszDatFile);
                    fclose(fpDat);
                    fpDat = NULL;
                    return -1;
                }
            }
        } else {
            printf("Cannot determine size of %s\n", gFecDefValue.pszDatFile);
            fclose(fpDat);
            fpDat = NULL;
            return -1;
        }
        fclose(fpDat);
        fpDat = NULL;
    }else {
        printf("Open %s failed\n", gFecDefValue.pszDatFile);
        return -1;
    }

    fpDat = fopen(gFecDefValue.pszComplexDatFile, "r");
    if (fpDat) {
        if (fstat(fileno(fpDat), &st) == 0) {
            tFecPtConfig.dwComplexCoeffDataSize = st.st_size;
            tFecPtConfig.pComplexCoeffData = calloc(1, st.st_size);
            if (tFecPtConfig.pComplexCoeffData) {
                if (fread(tFecPtConfig.pComplexCoeffData, tFecPtConfig.dwComplexCoeffDataSize, 1, fpDat) != 1) {
                    printf("Read %s failed\n", gFecDefValue.pszComplexDatFile);
                }
            }
        } else
            printf("Cannot determine complex size of %s\n", gFecDefValue.pszComplexDatFile);
        fclose(fpDat);
        fpDat = NULL;
    } else {
        printf("Open %s failed\n", gFecDefValue.pszComplexDatFile);
    }

    fpDat = fopen(gFecDefValue.pszMrfDatFile, "r");
    if (fpDat) {
        if (fstat(fileno(fpDat), &st) == 0) {
            tFecPtConfig.dwMrfCoeffDataSize = st.st_size;
            tFecPtConfig.pMrfCoeffData = calloc(1, st.st_size);
            if (tFecPtConfig.pMrfCoeffData) {
                if (fread(tFecPtConfig.pMrfCoeffData, tFecPtConfig.dwMrfCoeffDataSize, 1, fpDat) != 1) {
                    printf("Read %s failed\n", gFecDefValue.pszMrfDatFile);
                }
            }
        } else
            printf("Cannot determine mrf size of %s\n", gFecDefValue.pszMrfDatFile);
        fclose(fpDat);
        fpDat = NULL;
    } else {
        printf("Open %s failed\n", gFecDefValue.pszMrfDatFile);
    }

    tFecCellConfig.eFecMode = VMF_FEC_COEF_MODE_PT;
    tFecCellConfig.pFecConfig = &tFecPtConfig;

    setup_fec_layout(&tFecLytConfig, ptLayout, VMF_FEC_METHOD_CGE);

    //! setup layout
    VMF_FEC_LYT_Single(ptVsrcHandle, &tFecLytConfig, &tFecCellConfig);
    if (tFecPtConfig.pCoeffData) {
        free(tFecPtConfig.pCoeffData);
        tFecPtConfig.pCoeffData = NULL;
    }
    if (tFecPtConfig.pComplexCoeffData) {
        free(tFecPtConfig.pComplexCoeffData);
        tFecPtConfig.pComplexCoeffData = NULL;
    }
    if (tFecPtConfig.pMrfCoeffData) {
        free(tFecPtConfig.pMrfCoeffData);
        tFecPtConfig.pMrfCoeffData = NULL;
    }

    return 0;
}

size_t get_lens_curve_nodes(const char* filename, float* nodes_x, float* nodes_y)
{
    FILE *fd = fopen(filename, "r");
    if (fd == NULL) {
        printf("Unable to open %s.\n", filename);
        return 0;
    }

    char *lineptr = NULL;
    size_t buf_size = 0;
    ssize_t strlen;
    size_t element_idx = 0;
    char *token, *saveptr;
    char *temp_str;

    while ((strlen = getline(&lineptr, &buf_size, fd)) != -1) {
        if (strlen != 0) {
            lineptr[strlen - 1] = '\0';
            temp_str = lineptr;

            token = strtok_r(temp_str, ",", &saveptr);
            if (token == NULL) {
                printf("No ',' in line %ld \n", element_idx + 1);
                continue;
            }
            nodes_x[element_idx] = atof(token);
            nodes_y[element_idx] = atof(saveptr);
            element_idx++;
            if (element_idx >= 64) break;
        }
    }

    free(lineptr);
    fclose(fd);
    printf("Number of valid nodes: %ld\n", element_idx);
    return element_idx;
}

int loadCalibrateConfig(FEC_INIT_T* pFecInit, VMF_VSRC_FRONTEND_CONFIG_T *ptFrontConfig)
{
    struct stat info;
    //! calibrate
    if ((0 == stat(pFecInit->pszFecCalibratePath, &info)) && (info.st_mode & S_IFREG)) {
        dictionary* cal_ini = NULL;
        cal_ini = iniparser_load(pFecInit->pszFecCalibratePath);

        VMF_VSRC_FEC_INIT_CONFIG_T* fec_init = &ptFrontConfig->tFecInitConfig;
        memset(fec_init, 0, sizeof(VMF_VSRC_FEC_INIT_CONFIG_T));
        fec_init->fFecCenterOffsetX = iniparser_getint(cal_ini, "lens0:center_offset_x", 0);
        fec_init->fFecCenterOffsetY = iniparser_getint(cal_ini, "lens0:center_offset_y", 0);
        fec_init->fFecRadiusOffset = iniparser_getint(cal_ini, "lens0:radius_offset", 0);

        iniparser_freedict(cal_ini);
        return 0;
    }else
        return -1;
}

static inline void get_point_value(char* pcStr, unsigned int* pdwX, unsigned int* pdwY)
{
    char *pcToken, *pcRemainPtr;

    pcToken = strtok_r(pcStr, ",", &pcRemainPtr);
    if (pcToken == NULL) {
        printf("No ',' in line in string: %s \n", pcStr);
        return;
    }
    *pdwX = (unsigned int)atoi(pcToken);
    *pdwY = (unsigned int)atoi(pcRemainPtr);
}

static void get_extra_pt(dictionary* ini, const char* pre, Extra_PT* patExtra_PT, int dwCount)
{
    char pcSearchStr[64] = {0};
    const char* pcValueStr = NULL;
    Extra_PT* ptExtra_PT = NULL;
    int i = 0, j = 0;

    for (i = 0; i < dwCount; ++i) {
        ptExtra_PT = patExtra_PT + i;
        snprintf(pcSearchStr, 64, "%s:extra_pt_%d_en", pre, i);
        ptExtra_PT->bExtraPtEn = (unsigned int)iniparser_getint(ini, pcSearchStr, 0);
        for (j = 0; j < 4; ++j) {
            snprintf(pcSearchStr, 64, "%s:extra_pt_%d_%d", pre, i, j);
            pcValueStr = iniparser_getstring(ini, pcSearchStr, NULL);
            if (pcValueStr) {
                snprintf(pcSearchStr, 64, "%s", pcValueStr);
                get_point_value(pcSearchStr, &ptExtra_PT->dwX[j], &ptExtra_PT->dwY[j]);
            }
        }
    }
}

int loadFECConfig(FEC_INIT_T* pFecInit, VMF_VSRC_FRONTEND_CONFIG_T *ptFrontConfig)
{
    dictionary* ini = NULL;
    struct stat info;
    const char* tmp = NULL;

    //! check file
    if (0 != stat(pFecInit->pszFecConfPath, &info))
        return -1;

    if (!(info.st_mode & S_IFREG))
        return -1;
    ini = iniparser_load(pFecInit->pszFecConfPath);

    //! Get lens curve nodes
    if (ptFrontConfig) {
        VMF_VSRC_FEC_INIT_CONFIG_T *fec_init_config = NULL;
        fec_init_config = &(ptFrontConfig[0].tFecInitConfig);

        const char* tmp_nodes_path = iniparser_getstring(ini, "fec:user_define_lens_curve_nodes", NULL);
        fec_init_config->dwLensCurveNodeNum = get_lens_curve_nodes(tmp_nodes_path,
            fec_init_config->afLensCurveNodeX, fec_init_config->afLensCurveNodeY);
    }

    memset(&gFecDefValue, 0, sizeof(FECDefValue));

    gFecDefValue.lens_type = (VMF_FEC_LENS_TYPE)iniparser_getint(ini, "fec:lens_type", VMF_FEC_LENS_EQUIDISTANT);
    gFecDefValue.app_type = pFecInit->dwFecAppType;

    gFecDefValue.orig_zoom = (float)iniparser_getdouble(ini, "ori:zoom", 1.0);
    get_extra_pt(ini, "ori", &gFecDefValue.orig_extra_pt, 1);
    gFecDefValue.orig_src_radius = iniparser_getint(ini, "ori:src_radius", 0);
    gFecDefValue.orig_dst_radius = iniparser_getint(ini, "ori:dst_radius", 0);
    gFecDefValue.orig_convex_ratio_v = (float)iniparser_getdouble(ini, "ori:convex_ratio_v", 0.0);
    gFecDefValue.orig_convex_ratio_h = (float)iniparser_getdouble(ini, "ori:convex_ratio_h", 0.0);
    gFecDefValue.orig_depth_ratio = (float)iniparser_getdouble(ini, "ori:depth_ratio", 0.0);

    gFecDefValue.p180a_tilt  = (float)iniparser_getdouble(ini, "mode_p180a:tilt", 0.0);
    gFecDefValue.p180a_zoom  = (float)iniparser_getdouble(ini, "mode_p180a:zoom", 0.9);
    gFecDefValue.p180a_focal = (float)iniparser_getdouble(ini, "mode_p180a:focal", 0.63);
    gFecDefValue.p180a_spin = (float)iniparser_getdouble(ini, "mode_p180a:spin", 0.0);
    get_extra_pt(ini, "mode_p180a", &gFecDefValue.p180a_extra_pt, 1);

    gFecDefValue.p180o_tilt  = (float)iniparser_getdouble(ini, "mode_p180o:tilt", 0.0);
    gFecDefValue.p180o_zoom  = (float)iniparser_getdouble(ini, "mode_p180o:zoom",  0.93);
    gFecDefValue.p180o_focal = (float)iniparser_getdouble(ini, "mode_p180o:focal", 0.63);
    gFecDefValue.p180o_spin = (float)iniparser_getdouble(ini, "mode_p180o:spin", 0.0);
    get_extra_pt(ini, "mode_p180o", &gFecDefValue.p180o_extra_pt, 1);

    gFecDefValue.p180_dst_offset_x = (float)iniparser_getdouble(ini, "mode_p180:dst_offset_x", 0.0);
    gFecDefValue.p180_dst_offset_y = (float)iniparser_getdouble(ini, "mode_p180:dst_offset_y", 0.0);
    gFecDefValue.p180_dst_xy_ratio = (float)iniparser_getdouble(ini, "mode_p180:dst_xy_ratio", 1.0);

    gFecDefValue.m1r_tilt  = (float)iniparser_getdouble(ini, "mode_1r:tilt", 30);
    gFecDefValue.m1r_zoom  = (float)iniparser_getdouble(ini, "mode_1r:zoom", 1.0);
    gFecDefValue.m1r_focal = (float)iniparser_getdouble(ini, "mode_1r:focal", 0.63);
    gFecDefValue.m1r_pan_wall = (float)iniparser_getdouble(ini, "mode_1r:wall_pan", 0);
    gFecDefValue.m1r_tilt_wall = (float)iniparser_getdouble(ini, "mode_1r:wall_tilt", 0);
    gFecDefValue.m1r_dst_rotate = (float)iniparser_getdouble(ini, "mode_1r:dst_rotate", 0.0);
    get_extra_pt(ini, "mode_1r", &gFecDefValue.m1r_extra_pt, 1);

    //! PT mode
    tmp = iniparser_getstring(ini, "mode_pt:dat_path", NULL);
    if (tmp)
        gFecDefValue.pszDatFile = strdup(tmp);

    tmp = iniparser_getstring(ini, "mode_pt:complex_dat_path", NULL);
    if (tmp)
        gFecDefValue.pszComplexDatFile = strdup(tmp);

    tmp = iniparser_getstring(ini, "mode_pt:mrf_dat_path", NULL);
    if (tmp)
        gFecDefValue.pszMrfDatFile = strdup(tmp);

    //! EIS
    EIS_T* ptEis = &(gFecDefValue.tEis);
    tmp = iniparser_getstring(ini, "EIS:curve_nodes_path", NULL);
    if(tmp)
        ptEis->pszCurveNodesPath = strdup(tmp);
    ptEis->fGyroDataGain = (float)iniparser_getdouble(ini, "EIS:gyro_data_gain", 1.0);
    ptEis->dwGridSection = iniparser_getint(ini, "EIS:grid_section", 1);
    ptEis->dwMaxGridSection = iniparser_getint(ini, "EIS:max_grid_section", 40);
    ptEis->fCropRatio =  (float)iniparser_getdouble(ini, "EIS:crop_ratio", 0.0);
    ptEis->dwImageType = iniparser_getint(ini, "EIS:image_type", 0);
    ptEis->dwProcessMode = iniparser_getint(ini, "EIS:project_mode", 0);
    ptEis->dwCoordinateTransform[0] = iniparser_getint(ini, "EIS:coord_trans_0", 0);
    ptEis->dwCoordinateTransform[1] = iniparser_getint(ini, "EIS:coord_trans_1", 0);
    ptEis->dwCoordinateTransform[2] = iniparser_getint(ini, "EIS:coord_trans_2", 0);
    ptEis->sqwTimeOffset = (long long)iniparser_getint(ini, "EIS:time_offset", 0);
    ptEis->bImageRotate180 = iniparser_getint(ini, "EIS:image_rotate_180", 0);
    ptEis->sdwReadoutTimeOffset = iniparser_getint(ini, "EIS:readout_time_offset", 0);
    ptEis->fReadoutTimeRatio =  (float)iniparser_getdouble(ini, "EIS:readout_time_ratio", 1.0);
    ptEis->bForceOriRs = iniparser_getint(ini, "EIS:force_ori_resize", 0);

    //! Gyro daemon
    VMF_VSRC_GYRO_CONFIG_T* ptGyroConfig = &(gFecDefValue.tEis.tGyroConfig);
    ptGyroConfig->ulDeviceNum = iniparser_getint(ini, "gyro_daemon:device_num", 1);
    ptGyroConfig->dwSpeed = iniparser_getint(ini, "gyro_daemon:speed", 30);
    ptGyroConfig->dwFrequency = iniparser_getint(ini, "gyro_daemon:frequency", 200);
    ptGyroConfig->sdwGyroFsr = iniparser_getint(ini, "gyro_daemon:gyro_fsr", 250);
    ptGyroConfig->sdwAccelFsr = iniparser_getint(ini, "gyro_daemon:accel_fsr", -1);
    ptGyroConfig->dwSampleCount = iniparser_getint(ini, "gyro_daemon:sample_count", 200);
    tmp = iniparser_getstring(ini, "gyro_daemon:scm_pin_name", NULL);
    if(tmp)
        ptGyroConfig->pszPinName = strdup(tmp);

    iniparser_freedict(ini);
    return 0;
}

void free_fec_def_str(void){
    if (gFecDefValue.pszDatFile) {
        free(gFecDefValue.pszDatFile);
        gFecDefValue.pszDatFile = NULL;
    }

    if (gFecDefValue.pszComplexDatFile) {
        free(gFecDefValue.pszComplexDatFile);
        gFecDefValue.pszComplexDatFile = NULL;
    }

    if (gFecDefValue.pszMrfDatFile) {
        free(gFecDefValue.pszMrfDatFile);
        gFecDefValue.pszMrfDatFile = NULL;
    }
    /*
    if (g_tEis.pszCurveNodesPath) {
        free(g_tEis.pszCurveNodesPath);
        g_tEis.pszCurveNodesPath = NULL;
    }

    if (g_tEis.tGyroConfig.pszPinName) {
        free(g_tEis.tGyroConfig.pszPinName);
        g_tEis.tGyroConfig.pszPinName = NULL;
    }
    */
}

int setup_fec_mode(VMF_VSRC_HANDLE_T* ptVsrcHandle, VMF_LAYOUT_T* ptLayout, FEC_MODE mode, unsigned int dwEisEnable)
{
    switch (mode) {
        case FEC_MODE_1O:
            printf("[%s] FEC MODE: FEC_MODE_ORIGINAL\n", __func__);
            setup_fec_1o(ptVsrcHandle, ptLayout, dwEisEnable);
            break;

        case FEC_MODE_1R:
            printf("[%s] FEC MODE: FEC_MODE_1REGION\n", __func__);
            setup_fec_1r(ptVsrcHandle, ptLayout, dwEisEnable);
            break;

        case FEC_MODE_180_A:
            printf("[%s] FEC MODE: FEC_MODE_180_ALL_DIRECTION\n", __func__);
            setup_fec_p180a(ptVsrcHandle, ptLayout);
            break;

        case FEC_MODE_180_O:
            printf("[%s] FEC MODE: FEC_MODE_180_ONE_DIRECTION\n", __func__);
            setup_fec_p180o(ptVsrcHandle, ptLayout, dwEisEnable);
            break;

        case FEC_MODE_PT:
            printf("[%s] FEC MODE: FEC_PT_MODE\n", __func__);
            return setup_fec_pt_mode(ptVsrcHandle, ptLayout);

        default:
            printf("[%s] Invalid FEC MODE\n", __func__);
            return -1;
    }

    return 0;
}
