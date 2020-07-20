Shader "SimpNet/SimpNetL6"
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
            #include "SimpNetLayout.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

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
                    col.r = 1.0;
                }
                else if (insideArea(txW3BiasArea, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txSoftout1, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txSoftout2, px))
                {
                    col.r = 0.7;
                }

                return col;
            }
            ENDCG
        }
    }
}