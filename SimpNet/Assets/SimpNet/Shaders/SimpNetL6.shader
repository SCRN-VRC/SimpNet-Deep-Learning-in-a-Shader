﻿Shader "SimpNet/SimpNetL6"
{
    Properties
    {
        _Layer5 ("Layer 5", 2D) = "black" {}
        _L5Gradients ("Layer 6 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 6 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 6"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _Layer5;
            Texture2D<float3> _L6Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _Layer5_TexelSize;
            float4 _L6Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                col = _Time.y < 1.0 ? 0..xxx : col;

                [branch]
                if (insideArea(txW3Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
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
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
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
                        sum += _Layer5.Load(int3(txFC2a.xy + int2(0, j), 0)).x *
                            getW3(_FrameBuffer, int2(i, j));
                    }

                    sum += _FrameBuffer.Load(int3(txW3BiasArea.xy + int2(0, i), 0)).x;
                    col.r = sum;
                }
                else if (insideArea(txSoftout2, px))
                {
                    px -= txSoftout2.xy;
                    int i = px.y;

                    float sum = 0.0;
                    for (int j = 0; j < 12; j++) {
                        sum += exp(_FrameBuffer.Load(int3(txSoftout1.xy + int2(0, j), 0)).x);
                    }

                    col.r = exp(_FrameBuffer.Load(int3(txSoftout1.xy + int2(0, i), 0)).x) / sum;
                }

                return col;
            }
            ENDCG
        }
    }
}