class Nomenclature:

    DATE_MNEMONIC           = "Date"
    BIT_DEPTH_MNEMO         = "Bit Depth [ft]"
    DEPTH_MNEMONIC          = "Measured Depth [ft]"
    DEPTH_MNEMONIC_METRIC   = "Measured Depth [m]"
    TORTUOSITY_MNEMO        = "Tortuosity Idx"
    INCLINATION_MNEMO       = 'Inclination [°]'
    AZIMUTH_MNEMO           = 'Azimuth [°]'
    DLS_MNEMO               = 'DLS [°/100 ft]'
    GAMMA_RAY_LOG_MNEMO     = "Gamma Ray [API]"
    ROP_MNEMO               = "ROP [fph]"
    TORQUE_MNEMO            = "Torque [lb-ft]"
    WOB_MNEMO               = "Weight on Bit [klb]"
    STANDPIPE_PRESSURE_MNEMO="Standpipe Pressure [psi]"
    RPM_MNEMO               = "Surface Rotation [rpm]"
    HOOK_LOAD_MNEMO         = "Hook Load [klb]"
    HOLE_DEPTH_MNEMO        = "Hole Depth [ft]"
    FLOW_IN_MNEMO           = "Flow In [gpm]"
    BLOCK_POSITION_MNEMO    = "Block Position [ft]"
    RIG_STATE_MNEMO         = "Rig State"
    RIG_ACTIVITY_MNEMO      = "Activity"
    SECTION_PHASE_MNEMO     = "Section Operation Phase"
    TORTUOSITY_BIT_MNEMO    = 'Tortuosity Idx at Bit'
    FLEX_RIGIDITY_BIT_MNEMO = 'Flex Rigidity Difference at Bit'
    BLOCK_WEIGHT_MNEMO      = 'Block Weight [klb]'
    TRIP_OUT_MNEMO          = "Trip Out"
    DRILLING_MNEMO          = "Drilling"
    CIRCULATION_MNEMO       = "Circulation"
    BLOCK_WEIGHT_DELTA_MNEMO= "delta"
    BLOCK_WEIGHT_DELTA_DATES_MNEMO="dates"
    BLOCK_POSITION_TREND_MNEMO='Block Position Trend'
    FLOW_RATE_VARIABILITY_MNEMO='Flow Rate Variability'
    FLOW_RATE_MEAN_MNEMO        ='Flow Rate Mean'
    RPM_MEAN_MNEMO          = 'RPM Mean'
    HOOK_LOAD_MEAN_MNEMO    = 'Hook Load Mean'
    HOOK_LOAD_VARIABILITY_MNEMO='Hook Load Variability'
    EFF_HOOK_LOAD_MNEMO     = 'Effective Hook Load [klb]'
    ROP_MEAN_MNEMO          = 'ROP Mean'
    BACKREAMING_MNEMO       = 117
    TRIP_OUT_ELEV_MNEMO     = 112
    PUMPING_OUT_MNEMO       = 115
    REAMING_MNEMO           = 116
    DRILLING_ROT_MNEMO      = 119
    CONNECTION_MNEMO        = 118
    TRANSFORMER_MODEL_MNEMO = "transformer"
    LSTM_MODEL_MNEMO        = "lstm"
    GOAL_RIG_STATES         = [111,112,114,115,116,117,118,119,120,121]
    DICT_RIG_STATES         = {
                                    111:"Tripping in on elevators",
                                    112:"Tripping out on elevators",
                                    114:"Washing down",
                                    115:"Pumping out",
                                    116:"Reaming",
                                    117:"Backreaming",
                                    118:"Connection/Other surface operations",
                                    119:"Drilling (with surface rotation)",
                                    120:"Drilling (sliding)",
                                    121:"Circulating"
                                }
    CONVERTION_HASH_TABLE   = {
            "TIME":                             DATE_MNEMONIC,
            "DBTM":                             BIT_DEPTH_MNEMO,
            "BPOS":                             BLOCK_POSITION_MNEMO,
            "MFIA":                             FLOW_IN_MNEMO,
            "DMEA":                             HOLE_DEPTH_MNEMO,
            "HKLA":                             HOOK_LOAD_MNEMO,
            "RPMA":                             RPM_MNEMO,
            "SPPA":                             STANDPIPE_PRESSURE_MNEMO,
            "WOBA":                             WOB_MNEMO,
            "TQA":                              TORQUE_MNEMO,
            # "ROPA__":                           ROP_MNEMO,
            #Note (10/21/24): For some unknown issue within Shell's system, the 'ROPA__' channel
            # is sometimes unavailable (full with NaNs). An alternative is the one below.
            "ROPA__SHELL_CALCULATION_INPUT_TIME":ROP_MNEMO,
            "MD":                               DEPTH_MNEMONIC,
            "INCLINATION":                      INCLINATION_MNEMO,
            "AZIMUTH":                          AZIMUTH_MNEMO,
            "DLS":                              DLS_MNEMO,
            "DEPTH":                            DEPTH_MNEMONIC,
            "SGRC":                             GAMMA_RAY_LOG_MNEMO
        }

