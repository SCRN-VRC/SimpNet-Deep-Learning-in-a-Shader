Shader "SimpNet/SimpNetL3"
{
    Properties
    {
        _Layer2 ("Layer 2", 2D) = "black" {}
        _L3Gradients ("Layer 3 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 3 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 3"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "SimpNetLayout.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            Texture2D<float3> _Layer2;
            Texture2D<float3> _L3Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _Layer2_TexelSize;
            float4 _L3Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                col = _Time.y < 1.0 ? 0..xxx : col;

                [branch]
                if (insideArea(txKern3Area, px))
                {
                    col.r = 1.0;
                }
                else if (insideArea(txBias3Area, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txConv3Area, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txMax3Area, px))
                {
                    col.r = 0.7;
                }
                else if (insideArea(txiMax3Area, px))
                {
                    col.r = 0.6;
                }
                return col;
            }
            ENDCG
        }
    }
}