Shader "SimpNet/SimpNetL1"
{
    Properties
    {
        _CamIn ("Camera Input", 2D) = "black" {}
        _L1Gradients ("Layer 1 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 1 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 1"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "SimpNetLayout.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            Texture2D<float3> _CamIn;
            Texture2D<float3> _L1Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _CamIn_TexelSize;
            float4 _L1Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));

                [branch]
                if (insideArea(txKern1Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        col.r = rand(col.r) * 0.037;
                    }
                }
                else if (insideArea(txBias1Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        col.r = rand(col.r);
                    }
                }
                else if (insideArea(txConv1Area, px))
                {
                    px -= txConv1Area.xy;

                    float sum = 0.0;

                    int k = px.x % 32;
                    int i = px.y % 32;
                    int j = px.y / 8 + px.x / 32;
                    int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
                    int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

                    [unroll]
                    for (int l = 0; l < 3; l++) {
                        sum += _CamIn.Load(int3(i0, j0, 0))[l] * 1;
                    }


                    col.r = sum;

                }
                else if (insideArea(txMax1Area, px))
                {
                    col.r = 0.7;
                }
                else if (insideArea(txiMax1Area, px))
                {
                    col.r = 0.6;
                }
                return col;
            }
            ENDCG
        }
    }
}