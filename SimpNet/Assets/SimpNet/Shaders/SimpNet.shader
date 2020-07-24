﻿Shader "SimpNet/SimpNet"
{
    Properties
    {
        _CamIn ("Cam Input", 2D) = "black" {}
        _Buffer ("Buffer", 2D) = "black" {}
        _TargetClass ("Target Class #", Int) = 0
        _MaxDist ("Max Distance", Float) = 0.02
    }
    SubShader
    {
        Tags { "Queue"="Overlay+1" "ForceNoShadowCasting"="True" "IgnoreProjector"="True" }
        ZWrite Off
        ZTest Always
        Cull Off

        Pass
        {
            Lighting Off
            SeparateSpecular Off
            Fog { Mode Off }
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0

            #include "UnityCG.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"

            RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float4> _CamIn;
            Texture2D<float> _Buffer;
            float4 _Buffer_TexelSize;
            float _MaxDist;
            int _TargetClass;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float3 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = float4(v.uv * 2 - 1, 0, 1);
                #ifdef UNITY_UV_STARTS_AT_TOP
                v.uv.y = 1-v.uv.y;
                #endif
                o.uv.xy = UnityStereoTransformScreenSpaceTex(v.uv);
                o.uv.z = (distance(_WorldSpaceCameraPos,
                    mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz) > _MaxDist ||
                    !unity_OrthoParams.w) ?
                    -1 : 1;
                return o;
            }

            float frag (v2f i) : SV_Target
            {
                clip(i.uv.z);
                int2 px = _Buffer_TexelSize.zw * i.uv.xy;
                float col = _Buffer.Load(int3(px, 0)).x;

                [branch]
                if (insideArea(txL1, px))
                {
                    px -= txL1.xy;
                    [branch]
                    if (insideArea(txKern1Area, px))
                    {
                        px -= txKern1Area.xy;
                        int i = px.y % 3;
                        int j = px.x % 3;
                        int k = (px.y / 3) % 3;
                        int l = (px.x / 3) + (px.y / 9) * 8;
                        if (_Time.y < 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.037;
                            
                            // Debugging
                            col.r = i * j * k / (l + 1.0);
                        }
                    }
                    else if (insideArea(txBias1Area, px))
                    {
                        px -= txBias1Area.xy;
                        if (_Time.y < 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.5;

                            // Debugging
                            col.r = px.y / 32.0 - 0.5;
                        }
                    }
                    else if (insideArea(txConv1Area, px))
                    {
                        px -= txConv1Area.xy;

                        int i = px.y % 32;
                        int j = px.x % 32;
                        int k = px.x / 32 + (px.y / 32) * 4;
                        int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
                        int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

                        float sum = 0.0;
                        for (int l = 0; l < 3; l++) {
                            // sum += _CamIn.Load(int3(j0, i0, 0))[l] * getKern1(_Buffer, int4(0, 0, l, k));
                            // sum += _CamIn.Load(int3(j0, i1, 0))[l] * getKern1(_Buffer, int4(0, 1, l, k));
                            // sum += _CamIn.Load(int3(j0, i2, 0))[l] * getKern1(_Buffer, int4(0, 2, l, k));
                            // sum += _CamIn.Load(int3(j1, i0, 0))[l] * getKern1(_Buffer, int4(1, 0, l, k));
                            // sum += _CamIn.Load(int3(j1, i1, 0))[l] * getKern1(_Buffer, int4(1, 1, l, k));
                            // sum += _CamIn.Load(int3(j1, i2, 0))[l] * getKern1(_Buffer, int4(1, 2, l, k));
                            // sum += _CamIn.Load(int3(j2, i0, 0))[l] * getKern1(_Buffer, int4(2, 0, l, k));
                            // sum += _CamIn.Load(int3(j2, i1, 0))[l] * getKern1(_Buffer, int4(2, 1, l, k));
                            // sum += _CamIn.Load(int3(j2, i2, 0))[l] * getKern1(_Buffer, int4(2, 2, l, k));
                            
                            sum += testImage(i0, j0, l) * getKern1(_Buffer, int4(0, 0, l, k));
                            sum += testImage(i0, j1, l) * getKern1(_Buffer, int4(0, 1, l, k));
                            sum += testImage(i0, j2, l) * getKern1(_Buffer, int4(0, 2, l, k));
                            sum += testImage(i1, j0, l) * getKern1(_Buffer, int4(1, 0, l, k));
                            sum += testImage(i1, j1, l) * getKern1(_Buffer, int4(1, 1, l, k));
                            sum += testImage(i1, j2, l) * getKern1(_Buffer, int4(1, 2, l, k));
                            sum += testImage(i2, j0, l) * getKern1(_Buffer, int4(2, 0, l, k));
                            sum += testImage(i2, j1, l) * getKern1(_Buffer, int4(2, 1, l, k));
                            sum += testImage(i2, j2, l) * getKern1(_Buffer, int4(2, 2, l, k));
                        
                        }
                        sum += _Buffer.Load(int3(txL1.xy + txBias1Area.xy + int2(0, k), 0)).x;
                        col.r = actFn(sum);
                    }
                    else if (insideArea(txMax1Area, px))
                    {
                        px -= txMax1Area.xy;

                        int i = px.y % 16;
                        int j = px.x % 16;
                        int k = px.x / 16 + (px.y / 16) * 2;
                        int i0 = i * 2, i1 = i0 + 1;
                        int j0 = j * 2, j1 = j0 + 1;

                        float m = getConv1(_Buffer, int3(j0, i0, k));
                        m = max(m, getConv1(_Buffer, int3(j0, i1, k)));
                        m = max(m, getConv1(_Buffer, int3(j1, i0, k)));
                        m = max(m, getConv1(_Buffer, int3(j1, i1, k)));
                        col.r = m;
                    }
                    else if (insideArea(txiMax1Area, px))
                    {
                        px -= txiMax1Area.xy;

                        int i = px.y % 16;
                        int j = px.x % 16;
                        int k = px.x / 16 + (px.y / 16) * 2;
                        int i0 = i * 2, i1 = i0 + 1;
                        int j0 = j * 2, j1 = j0 + 1;

                        float bu;
                        float m = getConv1(_Buffer, int3(j0, i0, k));
                        col.r = i0 * 32 + j0;
                        
                        m = max(m, bu = getConv1(_Buffer, int3(j0, i1, k)));
                        col.r = (m == bu) ? (i1 * 32 + j0) : col.r;
                        
                        m = max(m, bu = getConv1(_Buffer, int3(j1, i0, k)));
                        col.r = (m == bu) ? (i0 * 32 + j1) : col.r;
                        
                        m = max(m, bu = getConv1(_Buffer, int3(j1, i1, k)));
                        col.r = (m == bu) ? (i1 * 32 + j1) : col.r;
                    }
                }
                else if (insideArea(txL2, px))
                {
                    px -= txL2.xy;
                    [branch]
                    if (insideArea(txKern2Area, px))
                    {
                        if (_Time.y < 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.003472;
                            
                            // Debugging
                            px -= txKern2Area.xy;
                            int i = px.y % 3;
                            int j = px.x % 3;
                            int k = (px.y / 3);
                            int l = (px.x / 3);
                            col.r = (i + j + k + l) / 1000.0;
                        }
                    }
                    else if (insideArea(txBias2Area, px))
                    {
                        if (_Time.y < 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.5;

                            // Debugging
                            px -= txBias2Area.xy;
                            col.r = 1.0 - (px.y / 64.0) - 0.5;
                        }
                    }
                    else if (insideArea(txConv2Area, px))
                    {
                        px -= txConv2Area.xy;
                        int i = px.y % 14;
                        int j = px.x % 14;
                        int k = px.x / 14 + (px.y / 14) * 8;
                        int i0 = i, i1 = i + 1, i2 = i + 2;
                        int j0 = j, j1 = j + 1, j2 = j + 2;

                        float sum = 0.0;
                        for (int l = 0; l < 32; l++) {
                            sum += getMax1(_Buffer, int3(j0, i0, l)) * getKern2(_Buffer, int4(0, 0, l, k));
                            sum += getMax1(_Buffer, int3(j0, i1, l)) * getKern2(_Buffer, int4(0, 1, l, k));
                            sum += getMax1(_Buffer, int3(j0, i2, l)) * getKern2(_Buffer, int4(0, 2, l, k));
                            sum += getMax1(_Buffer, int3(j1, i0, l)) * getKern2(_Buffer, int4(1, 0, l, k));
                            sum += getMax1(_Buffer, int3(j1, i1, l)) * getKern2(_Buffer, int4(1, 1, l, k));
                            sum += getMax1(_Buffer, int3(j1, i2, l)) * getKern2(_Buffer, int4(1, 2, l, k));
                            sum += getMax1(_Buffer, int3(j2, i0, l)) * getKern2(_Buffer, int4(2, 0, l, k));
                            sum += getMax1(_Buffer, int3(j2, i1, l)) * getKern2(_Buffer, int4(2, 1, l, k));
                            sum += getMax1(_Buffer, int3(j2, i2, l)) * getKern2(_Buffer, int4(2, 2, l, k));
                        }

                        sum += _Buffer.Load(int3(txL2.xy + txBias2Area.xy + int2(0, k), 0)).x;
                        col.r = actFn(sum);
                    }
                    else if (insideArea(txMax2Area, px))
                    {
                        px -= txMax2Area.xy;
                        int i = px.y % 7;
                        int j = px.x % 7;
                        int k = px.x / 7 + (px.y / 7) * 8;
                        int i0 = i * 2, i1 = i0 + 1;
                        int j0 = j * 2, j1 = j0 + 1;

                        float m = getConv2(_Buffer, int3(j0, i0, k));
                        m = max(m, getConv2(_Buffer, int3(j0, i1, k)));
                        m = max(m, getConv2(_Buffer, int3(j1, i0, k)));
                        m = max(m, getConv2(_Buffer, int3(j1, i1, k)));
                        col.r = m;
                    }
                    else if (insideArea(txiMax2Area, px))
                    {
                        px -= txiMax2Area.xy;
                        int i = px.y % 7;
                        int j = px.x % 7;
                        int k = px.x / 7 + (px.y / 7) * 8;
                        int i0 = i * 2, i1 = i0 + 1;
                        int j0 = j * 2, j1 = j0 + 1;

                        float bu;
                        float m = getConv2(_Buffer, int3(j0, i0, k));
                        col.r = i0 * 14 + j0;
                        
                        m = max(m, bu = getConv2(_Buffer, int3(j0, i1, k)));
                        col.r = (m == bu) ? (i1 * 14 + j0) : col.r;
                        
                        m = max(m, bu = getConv2(_Buffer, int3(j1, i0, k)));
                        col.r = (m == bu) ? (i0 * 14 + j1) : col.r;
                        
                        m = max(m, bu = getConv2(_Buffer, int3(j1, i1, k)));
                        col.r = (m == bu) ? (i1 * 14 + j1) : col.r;
                    }
                }
                else if (insideArea(txL3, px))
                {
                    px -= txL3.xy;
                    [branch]
                    if (insideArea(txKern3Area, px))
                    {
                        if (_Time.y < 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.0017361;
                            
                            // Debugging
                            px -= txKern3Area.xy;
                            int i = px.y % 3;
                            int j = px.x % 3;
                            int k = (px.y / 3);
                            int l = (px.x / 3);
                            col.r = (i + j) / float(k + l + 1.0);
                        }
                    }
                    else if (insideArea(txBias3Area, px))
                    {
                        if (_Time.y < 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.5;

                            // Debugging
                            px -= txBias3Area.xy;
                            col.r = (px.y / 128.0) - 0.5;
                        }
                    }
                    else if (insideArea(txConv3Area, px))
                    {
                        px -= txConv3Area.xy;
                        int i = px.y % 4;
                        int j = px.x % 4;
                        int k = px.y / 4;
                        int i0 = i * 2, i1 = i0 - 1, i2 = i0 + 1;
                        int j0 = j * 2, j1 = j0 - 1, j2 = j0 + 1;

                        // Padding
                        bool bi1 = i1 < 0, bj1 = j1 < 0, bi2 = i2 > 6, bj2 = j2 > 6;
                        bool b02 = bi1 || bj1, b03 = bi1 || bj2, b04 = bi2 || bj1, b05 = bi2 || bj2;

                        float sum = 0.0;
                        for (int l = 0; l < 64; l++) {
                            sum +=
                                (b02 ? 0.0 : getMax2(_Buffer, int3(j1, i1, l)) * getKern3(_Buffer, int4(0, 0, l, k))) +
                                (bi1 ? 0.0 : getMax2(_Buffer, int3(j0, i1, l)) * getKern3(_Buffer, int4(0, 1, l, k))) +
                                (b03 ? 0.0 : getMax2(_Buffer, int3(j2, i1, l)) * getKern3(_Buffer, int4(0, 2, l, k))) +
                                (bj1 ? 0.0 : getMax2(_Buffer, int3(j1, i0, l)) * getKern3(_Buffer, int4(1, 0, l, k))) +
                                getMax2(_Buffer, int3(j0, i0, l)) * getKern3(_Buffer, int4(1, 1, l, k)) +
                                (bj2 ? 0.0 : getMax2(_Buffer, int3(j2, i0, l)) * getKern3(_Buffer, int4(1, 2, l, k))) +
                                (b04 ? 0.0 : getMax2(_Buffer, int3(j1, i2, l)) * getKern3(_Buffer, int4(2, 0, l, k))) +
                                (bi2 ? 0.0 : getMax2(_Buffer, int3(j0, i2, l)) * getKern3(_Buffer, int4(2, 1, l, k))) +
                                (b05 ? 0.0 : getMax2(_Buffer, int3(j2, i2, l)) * getKern3(_Buffer, int4(2, 2, l, k)));
                        }
                        sum += _Buffer.Load(int3(txL3.xy + txBias3Area.xy + int2(0, k), 0)).x;
                        col.r = actFn(sum);
                    }
                    else if (insideArea(txMax3Area, px))
                    {
                        px -= txMax3Area.xy;
                        int i = px.y % 2;
                        int j = px.x % 2;
                        int k = px.x / 2 + (px.y / 2);
                        int i0 = i * 2, i1 = i0 + 1;
                        int j0 = j * 2, j1 = j0 + 1;

                        float m = getConv3(_Buffer, int3(j0, i0, k));
                        m = max(m, getConv3(_Buffer, int3(j0, i1, k)));
                        m = max(m, getConv3(_Buffer, int3(j1, i0, k)));
                        m = max(m, getConv3(_Buffer, int3(j1, i1, k)));
                        col.r = m;
                    }
                    else if (insideArea(txiMax3Area, px))
                    {
                        px -= txiMax3Area.xy;
                        int i = px.y % 2;
                        int j = px.x % 2;
                        int k = px.x / 2 + (px.y / 2);
                        int i0 = i * 2, i1 = i0 + 1;
                        int j0 = j * 2, j1 = j0 + 1;

                        float bu;
                        float m = getConv3(_Buffer, int3(j0, i0, k));
                        col.r = i0 * 4 + j0;
                        
                        m = max(m, bu = getConv3(_Buffer, int3(j0, i1, k)));
                        col.r = (m == bu) ? (i1 * 4 + j0) : col.r;
                        
                        m = max(m, bu = getConv3(_Buffer, int3(j1, i0, k)));
                        col.r = (m == bu) ? (i0 * 4 + j1) : col.r;
                        
                        m = max(m, bu = getConv3(_Buffer, int3(j1, i1, k)));
                        col.r = (m == bu) ? (i1 * 4 + j1) : col.r;
                    }
                }
                else if (insideArea(txL4, px))
                {
                    px -= txL4.xy;
                    [branch]
                    if (insideArea(txW1Area, px))
                    {
                        if (_Time.y <= 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.001953125;
                            
                            // Debugging
                            px -= txW1Area.xy;
                            int i = px.y % 2;
                            int j = px.x % 2;
                            int k = (px.y / 2);
                            int l = (px.x / 2);
                            col.r = (i * (j + i)) * k / float(l * k + 1);
                        }
                    }
                    else if (insideArea(txW1BiasArea, px))
                    {
                        if (_Time.y <= 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.5;

                            // Debugging
                            px -= txW1BiasArea.xy;
                            col.r = (px.y % 8) / 8.0;
                        }
                    }
                    else if (insideArea(txFC1s, px))
                    {
                        px -= txFC1s.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int k = 0; k < 2; k++) {
                            for (int l = 0; l < 2; l++) {
                                for (int j = 0; j < 128; j++) {
                                    sum += getMax3(_Buffer, int3(l, k, j)) * getW1(_Buffer, int4(k, l, j, i));
                                }
                            }
                        }
                        sum += _Buffer.Load(int3(txL4.xy + txW1BiasArea.xy + int2(0, i), 0)).x;
                        col.r = sum;
                    }
                    else if (insideArea(txFC1a, px))
                    {
                        px -= txFC1a.xy;
                        col.r = actFn(_Buffer.Load(int3(txL4.xy + txFC1s.xy + int2(0, px.y), 0)).x);
                    }
                }
                else if (insideArea(txL5, px))
                {
                    px -= txL5.xy;
                    [branch]
                    if (insideArea(txW2Area, px))
                    {
                        if (_Time.y <= 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.0078125;
                            
                            // Debugging
                            px -= txW2Area.xy;
                            int i = px.y;
                            int j = px.x;
                            col.r = (i + j) / (128.0 * 128.0);
                        }
                    }
                    else if (insideArea(txW2BiasArea, px))
                    {
                        if (_Time.y <= 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.5;

                            // Debugging
                            px -= txW2BiasArea.xy;
                            col.r = 1.0 / (px.y + 1.0);
                        }
                    }
                    else if (insideArea(txFC2s, px))
                    {
                        px -= txFC2s.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int j = 0; j < 128; j++) {
                            sum += _Buffer.Load(int3(txL4.xy + txFC1a.xy + int2(0, j), 0)).x *
                                _Buffer.Load(int3(txL5.xy + txW2Area.xy + int2(i, j), 0)).x;
                        }
                        sum += _Buffer.Load(int3(txL5.xy + txW2BiasArea.xy + int2(0, i), 0)).x;
                        col.r = sum;
                    }
                    else if (insideArea(txFC2a, px))
                    {
                        px -= txFC2a.xy;
                        col.r = actFn(_Buffer.Load(int3(txL5.xy + txFC2s.xy + int2(0, px.y), 0)).x);
                    }
                }
                else if (insideArea(txL6, px))
                {
                    px -= txL6.xy;
                    [branch]
                    if (insideArea(txW3Area, px))
                    {
                        if (_Time.y <= 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.0078125;
                            
                            // Debugging
                            px -= txW3Area.xy;
                            int i = px.y + (px.x / 12) * 64;
                            int j = px.x % 12;
                            col.r = (i + j) / 100000000.0;
                        }
                    }
                    else if (insideArea(txW3BiasArea, px))
                    {
                        if (_Time.y <= 1.0)
                        {
                            // col.r = px.y * _Buffer_TexelSize.z + px.x;
                            // col.r = rand(col.r) * 0.5;

                            // Debugging
                            px -= txW3BiasArea.xy;
                            col.r = 1.0 - (px.y / 12.0);
                        }
                    }
                    else if (insideArea(txSoftout1, px))
                    {
                        px -= txSoftout1.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int j = 0; j < 128; j++) {
                            sum += _Buffer.Load(int3(txL5.xy + txFC2a.xy + int2(0, j), 0)).x *
                                getW3(_Buffer, int2(i, j));
                        }

                        sum += _Buffer.Load(int3(txL6.xy + txW3BiasArea.xy + int2(0, i), 0)).x;
                        col.r = sum;
                    }
                    else if (insideArea(txSoftout2, px))
                    {
                        px -= txSoftout2.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int j = 0; j < 12; j++) {
                            sum += exp(_Buffer.Load(int3(txL6.xy + txSoftout1.xy + int2(0, j), 0)).x);
                        }

                        col.r = exp(_Buffer.Load(int3(txL6.xy + txSoftout1.xy + int2(0, i), 0)).x) / sum;
                    }
                }
                else if (insideArea(txB1, px))
                {
                    px -= txB1.xy;
                    [branch]
                    if (insideArea(txDBW3Area, px))
                    {
                        px -= txDBW3Area.xy;
                        int i = px.y;
                        col.r = _Buffer.Load(int3(txL6.xy + txSoftout2.xy + int2(0, i), 0)).x - (i == _TargetClass ? 1.0 : 0.0);
                    }
                    else if (insideArea(txDW3Area, px))
                    {
                        px -= txDW3Area.xy;
                        int i = px.y;
                        int j = px.x;
                        col.r = _Buffer.Load(int3(txB1.xy + txDBW3Area.xy + int2(0, j), 0)).x *
                            _Buffer.Load(int3(txL5.xy + txFC2a.xy + int2(0, i), 0)).x;
                    }
                    else if (insideArea(txDBW2Area, px))
                    {
                        px -= txDW2Area.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int k = 0; k < 12; k++) {
                            sum += _Buffer.Load(int3(txB1.xy + txDBW3Area.xy + int2(0, k), 0)).x *
                                getW3(_Buffer, int2(k, i));
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txDW2Area, px))
                    {
                        px -= txDW2Area.xy;
                        int i = px.y;
                        int j = px.x;

                        col.r = _Buffer.Load(int3(txB1.xy + txDBW2Area.xy + int2(0, i), 0)).x *
                            dactFn(_Buffer.Load(int3(txL5.xy + txFC2s.xy + int2(0, i), 0)).x) *
                            _Buffer.Load(int3(txL4.xy + txFC1a.xy + int2(0, j), 0)).x;
                    }
                    else if (insideArea(txDBW1Area, px))
                    {
                        px -= txDBW1Area.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int k = 0; k < 128; k++) {
                            sum += _Buffer.Load(int3(txB1.xy + txDBW2Area.xy + int2(0, k), 0)).x *
                                dactFn(_Buffer.Load(int3(txL5.xy + txFC2s.xy + int2(0, k), 0)).x) *
                                _Buffer.Load(int3(txL5.xy + txW2Area.xy + int2(i, k), 0)).x;
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txDW1Area, px))
                    {
                        px -= txDW1Area.xy;
                        int i = px.y % 2;
                        int j = px.x % 2;
                        int k = px.y / 2;
                        int l = px.x / 2;

                        col.r = _Buffer.Load(int3(txB1.xy + txDBW1Area.xy + int2(0, l), 0)).x *
                            dactFn(_Buffer.Load(int3(txL4.xy + txFC1s.xy + int2(0, l), 0)).x) *
                            getMax3(_Buffer, int3(j, i, k));
                    }
                }
                else if (insideArea(txB2, px))
                {
                    px -= txB2.xy;
                    [branch]
                    if (insideArea(txEMax3Area, px))
                    {
                        px -= txEMax3Area.xy;
                        int i = px.y % 2;
                        int j = px.x % 2;
                        int k = px.y / 2;

                        float sum = 0.0;
                        for (int l = 0; l < 128; l++) {
                            sum += _Buffer.Load(int3(txB1.xy + txDBW1Area + int2(0, l), 0)).x *
                                dactFn(_Buffer.Load(int3(txL4.xy + txFC1s.xy + int2(0, l), 0)).x) *
                                getW1(_Buffer, int4(i, j, k, l));
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txDB3Area, px))
                    {
                        px -= txDB3Area.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int j = 0; j < 2; j++) {
                            for (int k = 0; k < 2; k++) {
                                sum += getEMax3(_Buffer, int3(j, k, i));
                            }
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txEConv3Area, px))
                    {
                        px -= txEConv3Area.xy;
                        int i = px.y % 4;
                        int j = px.x % 4;
                        int k = px.y / 4;
                        int i0 = i / 2, j0 = j / 2;

                        col.r = abs(getIMax3(_Buffer, int3(j0, i0, k)) - float(i * 4 + j)) < eps ?
                            getEMax3(_Buffer, int3(j0, i0, k)) : 0.0;
                    }
                    else if (insideArea(txDiConv3Area, px))
                    {
                        px -= txDiConv3Area.xy;
                        int i = px.y % 7;
                        int j = px.x % 7;
                        int k = px.x / 7 + (px.y / 7) * 8;
                        int i0 = i / 2, j0 = j / 2;
                        
                        col.r = ((i % 2 == 1) || (j % 2 == 1)) ? 0.0 : getEConv3(_Buffer, int3(j0, i0, k));
                    }
                    else if (insideArea(txDKern3Area, px))
                    {
                        px -= txDKern3Area.xy;
                        int i = px.y % 3;
                        int j = px.x % 3;
                        int k = px.x / 3;
                        int l = px.y / 3;

                        float sum = 0.0;
                        for (int x = 0; x < 7; x++) {
                            for (int y = 0; y < 7; y++) {
                                int l2x = x + i - 1;
                                int l2y = y + j - 1;
                                // Padding
                                bool b = l2x < 0 || l2y < 0 || l2x > 6 || l2y > 6;
                                sum += b ? 0.0 : getMax2(_Buffer, int3(l2y, l2x, k)) * getDiConv3(_Buffer, int3(y, x, l));
                            }
                        }
                        col.r = sum;
                    }
                }
                else if (insideArea(txB3, px))
                {
                    px -= txB3.xy;
                    [branch]
                    if (insideArea(txEMax2Area, px))
                    {
                        px -= txEMax2Area.xy;
                        int i = px.y % 7;
                        int j = px.x % 7;
                        int k = px.x / 7 + (px.y / 7) * 8;
                        int i0 = i, i1 = i - 1, i2 = i + 1;
                        int j0 = j, j1 = j - 1, j2 = j + 1;
                        // Padding
                        bool b0 = (i1 < 0 || j1 < 0), b1 = (i1 < 0), b2 = (i1 < 0 || j2 > 6), b3 = (j1 < 0);
                        bool b4 = (j2 > 6), b5 = (i2 > 6 || j1 < 0), b6 = (i2 > 6), b7 = (i2 > 6 || j2 > 6);

                        float sum = 0.0;
                        for (int l = 0; l < 128; l++) {
                            sum += (b0 ? 0.0 : getDiConv3(_Buffer, int3(j1, i1, l)) * getKern3(_Buffer, int4(2, 2, k, l)));
                            sum += (b1 ? 0.0 : getDiConv3(_Buffer, int3(j0, i1, l)) * getKern3(_Buffer, int4(2, 1, k, l)));
                            sum += (b2 ? 0.0 : getDiConv3(_Buffer, int3(j2, i1, l)) * getKern3(_Buffer, int4(2, 0, k, l)));
                            sum += (b3 ? 0.0 : getDiConv3(_Buffer, int3(j1, i0, l)) * getKern3(_Buffer, int4(1, 2, k, l)));
                            sum += getDiConv3(_Buffer, int3(j0, i0, l)) * getKern3(_Buffer, int4(1, 1, k, l));
                            sum += (b4 ? 0.0 : getDiConv3(_Buffer, int3(j2, i0, l)) * getKern3(_Buffer, int4(1, 0, k, l)));
                            sum += (b5 ? 0.0 : getDiConv3(_Buffer, int3(j1, i2, l)) * getKern3(_Buffer, int4(0, 2, k, l)));
                            sum += (b6 ? 0.0 : getDiConv3(_Buffer, int3(j0, i2, l)) * getKern3(_Buffer, int4(0, 1, k, l)));
                            sum += (b7 ? 0.0 : getDiConv3(_Buffer, int3(j2, i2, l)) * getKern3(_Buffer, int4(0, 0, k, l)));
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txDB2Area, px))
                    {
                        px -= txDB2Area.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int j = 0; j < 7; j++) {
                            for (int k = 0; k < 7; k++) {
                                sum += getEMax2(_Buffer, int3(j, k, i));
                            }
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txEConv2Area, px))
                    {
                        px -= txEConv2Area.xy;
                        int i = px.y % 14;
                        int j = px.x % 14;
                        int k = px.x / 14 + (px.y / 14) * 8;
                        int i0 = i / 2, j0 = j / 2;

                        col.r = abs(getIMax2(_Buffer, int3(j0, i0, k)) - float(i * 14 + j)) < eps ?
                            getEMax2(_Buffer, int3(j0, i0, k)) : 0.0;
                    }
                    else if (insideArea(txDKern2Area, px))
                    {
                        px -= txDKern2Area.xy;
                        int i = px.y % 3;
                        int j = px.x % 3;
                        int k = px.x / 3;
                        int l = px.y / 3;

                        float sum = 0.0;
                        for (int x = 0; x < 14; x++) {
                            for (int y = 0; y < 14; y++) {
                                int l1x = x + i;
                                int l1y = y + j;
                                sum += getMax1(_Buffer, int3(l1y, l1x, k)) * getEConv2(_Buffer, int3(y, x, l));
                            }
                        }
                        col.r = sum;
                    }
                }
                else if (insideArea(txB4, px))
                {
                    px -= txB4.xy;
                    [branch]
                    if (insideArea(txPConv2Area, px))
                    {
                        px -= txPConv2Area.xy;
                        int i = px.y % 18;
                        int j = px.x % 18;
                        int k = px.x / 18 + (px.y / 18) * 8;

                        col.r = i < 2 || j < 2 || i > 15 || j > 15 ? 0.0 : getEConv2(_Buffer, int3(j - 2, i - 2, k));
                    }
                    else if (insideArea(txEMax1Area, px))
                    {
                        px -= txEMax1Area.xy;
                        int i = px.y % 16;
                        int j = px.x % 16;
                        int k = px.x / 16 + (px.y / 16) * 4;

                        float sum = 0.0;
                        for (int l = 0; l < 64; l++) {
                            sum += getPConv2(_Buffer, int3(j + 0, i + 0, l)) * getKern2(_Buffer, int4(2, 2, k, l));
                            sum += getPConv2(_Buffer, int3(j + 1, i + 0, l)) * getKern2(_Buffer, int4(2, 1, k, l));
                            sum += getPConv2(_Buffer, int3(j + 2, i + 0, l)) * getKern2(_Buffer, int4(2, 0, k, l));
                            sum += getPConv2(_Buffer, int3(j + 0, i + 1, l)) * getKern2(_Buffer, int4(1, 2, k, l));
                            sum += getPConv2(_Buffer, int3(j + 1, i + 1, l)) * getKern2(_Buffer, int4(1, 1, k, l));
                            sum += getPConv2(_Buffer, int3(j + 2, i + 1, l)) * getKern2(_Buffer, int4(1, 0, k, l));
                            sum += getPConv2(_Buffer, int3(j + 0, i + 2, l)) * getKern2(_Buffer, int4(0, 2, k, l));
                            sum += getPConv2(_Buffer, int3(j + 1, i + 2, l)) * getKern2(_Buffer, int4(0, 1, k, l));
                            sum += getPConv2(_Buffer, int3(j + 2, i + 2, l)) * getKern2(_Buffer, int4(0, 0, k, l));
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txDB1Area, px))
                    {
                        px -= txDB1Area.xy;
                        int i = px.y;

                        float sum = 0.0;
                        for (int j = 0; j < 16; j++) {
                            for (int k = 0; k < 16; k++) {
                                sum += getEMax1(_Buffer, int3(j, k, i));
                            }
                        }
                        col.r = sum;
                    }
                    else if (insideArea(txEConv1Area, px))
                    {
                        px -= txEConv1Area;
                        int i = px.y % 32;
                        int j = px.x % 32;
                        int k = px.x / 32 + (px.y / 32) * 4;
                        int i0 = i / 2, j0 = j / 2;
                        
                        col.r = abs(getIMax1(_Buffer, int3(j0, i0, k)) - float(i * 32 + j)) < eps ?
                            getEMax1(_Buffer, int3(j0, i0, k)) : 0.0;
                    }
                    else if (insideArea(txDiConv1Area, px))
                    {
                        px -= txDiConv1Area.xy;
                        int i = px.y % 63;
                        int j = px.x % 63;
                        int k = px.x / 63 + (px.y / 63) * 4;
                        int i0 = i / 2, j0 = j / 2;

                        col.r = ((i % 2 == 1) || (j % 2 == 1)) ? 0.0 : getEConv1(_Buffer, int3(j0, i0, k));
                    }
                    else if (insideArea(txDKern1Area, px))
                    {
                        px -= txDKern1Area.xy;
                        int i = px.y % 3;
                        int j = px.x % 3;
                        int k = (px.y / 3) % 3;
                        int l = px.x / 3 + (px.y / 9) * 8;
                        
                        float sum = 0.0;
                        for (int x = 0; x < 63; x++) {
                            for (int y = 0; y < 63; y++) {
                                int l1x = x + i;
                                int l1y = y + j;
                                sum += testImage(l1x, l1y, k) * getDiConv1(_Buffer, int3(y, x, l));
                            }
                        }
                        col.r = sum;
                    }
                }

                return col;
            }
            ENDCG
        }
    }
}