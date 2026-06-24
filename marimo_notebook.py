# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "datasets==5.0.0",
#     "matplotlib==3.11.0",
#     "pandas==3.0.3",
#     "wanderland==0.1.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="full",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import gc
    import json
    import random
    import re
    import subprocess
    import threading

    import marimo as mo
    import pandas as pd
    import torch
    import wanderland as bp
    from wanderland import move_forward, turn_left, turn_right, collect_gem
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        TextIteratorStreamer,
        set_seed,
    )

    # Paper-reference card: narrower + centered, title links to alphaXiv, theme-aware, no image.
    paper_card = mo.Html(r"""
    <div style="max-width:645px;margin:0 auto;border:1px solid rgba(124,140,64,0.35);border-radius:10px;padding:16px 18px;background:rgba(124,140,64,0.06);box-shadow:0 8px 24px rgba(0,0,0,0.05);">
      <a href="https://www.alphaxiv.org/abs/2505.12135" style="display:block;font-size:1.05rem;line-height:1.33;font-weight:800;color:inherit;text-decoration:none;">LLM-BabyBench: Understanding and Evaluating Grounded Planning and Reasoning in LLMs &#8599;</a>
      <div style="margin-top:8px;font-size:0.9rem;line-height:1.4;opacity:0.75;">Omar Choukrani<sup>*</sup> &middot; Idriss Malek<sup>*</sup> &middot; Daniil Orel<sup>*</sup> &middot; Zhuohan Xie &middot; Zangir Iklassov &middot; Martin Tak&aacute;&ccaron; &middot; Salem Lahlou</div>
      <div style="margin-top:4px;font-size:0.8rem;opacity:0.55;">MBZUAI, Abu Dhabi &middot; NeurIPS 2025</div>
      <div style="margin-top:13px;display:flex;gap:16px;flex-wrap:wrap;">
        <a href="https://github.com/choukrani/llm-babybench" style="color:#7c8c40;font-size:0.88rem;font-weight:750;text-decoration:none;">GitHub</a>
        <a href="https://huggingface.co/datasets/salem-mbzuai/LLM-BabyBench" style="color:#7c8c40;font-size:0.88rem;font-weight:750;text-decoration:none;">HuggingFace</a>
      </div>
    </div>
    """)

    """Mo the Mossball — drop-in animated ASCII chapter banners.

        import marimo as mo
        from mo_banner_widgets import banner, BANNERS

        banner("meet_mo")            # mo.Html, with the section heading
        banner("twist", title=False) # just the visual
        list(BANNERS)                # available banner keys
    """

    # key -> (heading, self-contained HTML with namespaced CSS animations)
    BANNERS = {
        'title': (
            'Can LLMs Plan?',
            r"""<style>@keyframes title_moBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}@keyframes title_moBlink{0%,93%{opacity:0}94%,96%{opacity:1}97%,100%{opacity:0}}@keyframes title_moPulse{0%,100%{opacity:.55}50%{opacity:1}}</style><div style="display:flex;align-items:center;justify-content:center;gap:70px;width:100%;box-sizing:border-box;background:transparent;padding:20px 16px;border-radius:10px;overflow:hidden"><span style="position:relative;display:inline-block;line-height:0"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">                          <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                              <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        </pre><span style="position:absolute;top:0;left:0;animation:title_moBlink 4.3s steps(1,end) infinite"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">                          <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                              <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        </pre></span></span><span style="display:inline-block;animation:title_moBob 2.4s ease-in-out 0.0s infinite;margin-top:0px;will-change:transform"><span style="display:inline-block;animation:title_moPulse 1.6s ease-in-out 0.0s infinite"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"> <span style="color:rgb(255,242,180)">███████</span> 
    <span style="color:rgb(255,242,180)">██</span><span style="color:rgb(245,205,70)">█████</span><span style="color:rgb(255,242,180)">██</span>
    <span style="color:rgb(255,242,180)">█</span><span style="color:rgb(245,205,70)">█</span><span style="color:rgb(150,110,20)">█</span>   <span style="color:rgb(150,110,20)">█</span><span style="color:rgb(245,205,70)">█</span><span style="color:rgb(255,242,180)">█</span>
    <span style="color:rgb(255,242,180)">█</span><span style="color:rgb(245,205,70)">█</span><span style="color:rgb(150,110,20)">█</span>  <span style="color:rgb(150,110,20)">█</span><span style="color:rgb(245,205,70)">██</span><span style="color:rgb(255,242,180)">█</span>
         <span style="color:rgb(150,110,20)">█</span><span style="color:rgb(245,205,70)">██</span><span style="color:rgb(255,242,180)">█</span>
        <span style="color:rgb(150,110,20)">█</span><span style="color:rgb(245,205,70)">██</span>  
       <span style="color:rgb(150,110,20)">█</span><span style="color:rgb(245,205,70)">██</span>   
      <span style="color:rgb(150,110,20)">█</span><span style="color:rgb(245,205,70)">██</span>    
      <span style="color:rgb(245,205,70)">███</span>    
      <span style="color:rgb(150,110,20)">███</span>    

      <span style="color:rgb(245,205,70)">███</span>    
      <span style="color:rgb(150,110,20)">███</span>    </pre></span></span></div>""",
        ),
        'meet_mo': (
            'Act 1 · Meet Mo',
            r"""<style>@keyframes meet_mo_moBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}@keyframes meet_mo_moBlink{0%,93%{opacity:0}94%,96%{opacity:1}97%,100%{opacity:0}}</style><div style="display:flex;align-items:center;justify-content:center;gap:56px;width:100%;box-sizing:border-box;background:transparent;padding:20px 16px;border-radius:10px;overflow:hidden"><span style="position:relative;display:inline-block;line-height:0"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">                          <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                              <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        </pre><span style="position:absolute;top:0;left:0;animation:meet_mo_moBlink 4.3s steps(1,end) infinite"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">                          <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                              <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        </pre></span></span><div style="display:flex;align-items:center;gap:34px"><span style="display:inline-block;animation:meet_mo_moBob 2.0s ease-in-out 0.0s infinite;margin-top:-6px;will-change:transform;"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">   <span style="color:rgb(18,95,72)">██████</span>   
     <span style="color:rgb(18,95,72)">██</span><span style="color:rgb(120,235,180)">██████</span><span style="color:rgb(18,95,72)">██</span> 
    <span style="color:rgb(18,95,72)">██</span><span style="color:rgb(120,235,180)">█</span><span style="color:rgb(205,255,235)">██████</span><span style="color:rgb(120,235,180)">█</span><span style="color:rgb(18,95,72)">██</span>
    <span style="color:rgb(18,95,72)">█</span><span style="color:rgb(120,235,180)">█</span><span style="color:rgb(55,200,140)">████████</span><span style="color:rgb(120,235,180)">█</span><span style="color:rgb(18,95,72)">█</span>
    <span style="color:rgb(18,95,72)">█</span><span style="color:rgb(28,150,100)">█</span><span style="color:rgb(55,200,140)">████████</span><span style="color:rgb(28,150,100)">█</span><span style="color:rgb(18,95,72)">█</span>
     <span style="color:rgb(18,95,72)">█</span><span style="color:rgb(28,150,100)">█</span><span style="color:rgb(55,200,140)">██████</span><span style="color:rgb(28,150,100)">█</span><span style="color:rgb(18,95,72)">█</span> 
     <span style="color:rgb(18,95,72)">█</span><span style="color:rgb(28,150,100)">██</span><span style="color:rgb(55,200,140)">████</span><span style="color:rgb(28,150,100)">██</span><span style="color:rgb(18,95,72)">█</span> 
      <span style="color:rgb(18,95,72)">█</span><span style="color:rgb(28,150,100)">██</span><span style="color:rgb(55,200,140)">██</span><span style="color:rgb(28,150,100)">██</span><span style="color:rgb(18,95,72)">█</span>  
      <span style="color:rgb(18,95,72)">██</span><span style="color:rgb(28,150,100)">████</span><span style="color:rgb(18,95,72)">██</span>  
       <span style="color:rgb(18,95,72)">██</span><span style="color:rgb(28,150,100)">██</span><span style="color:rgb(18,95,72)">██</span>   
        <span style="color:rgb(18,95,72)">█</span><span style="color:rgb(28,150,100)">██</span><span style="color:rgb(18,95,72)">█</span>    
         <span style="color:rgb(18,95,72)">██</span>     </pre></span><span style="display:inline-block;animation:meet_mo_moBob 2.6s ease-in-out -0.7s infinite;margin-top:12px;will-change:transform;"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">   <span style="color:rgb(255,242,180)">████</span>   
     <span style="color:rgb(255,242,180)">█</span><span style="color:rgb(245,210,80)">██████</span><span style="color:rgb(255,242,180)">█</span> 
    <span style="color:rgb(255,242,180)">█</span><span style="color:rgb(245,210,80)">█</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(120,90,15)">████</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(245,210,80)">█</span><span style="color:rgb(255,242,180)">█</span>
    <span style="color:rgb(245,210,80)">█</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(120,90,15)">█</span>    <span style="color:rgb(120,90,15)">█</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(245,210,80)">█</span>
    <span style="color:rgb(245,210,80)">█</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(120,90,15)">█</span>    <span style="color:rgb(120,90,15)">█</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(245,210,80)">█</span>
    <span style="color:rgb(255,242,180)">█</span><span style="color:rgb(245,210,80)">█</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(120,90,15)">████</span><span style="color:rgb(185,145,30)">█</span><span style="color:rgb(245,210,80)">█</span><span style="color:rgb(255,242,180)">█</span>
     <span style="color:rgb(255,242,180)">█</span><span style="color:rgb(245,210,80)">██████</span><span style="color:rgb(255,242,180)">█</span> 
       <span style="color:rgb(255,242,180)">████</span>   </pre></span><span style="display:inline-block;animation:meet_mo_moBob 2.3s ease-in-out -1.3s infinite;margin-top:-16px;will-change:transform;"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">   <span style="color:rgb(18,80,110)">██████</span>   
     <span style="color:rgb(18,80,110)">██</span><span style="color:rgb(110,210,235)">██████</span><span style="color:rgb(18,80,110)">██</span> 
    <span style="color:rgb(18,80,110)">██</span><span style="color:rgb(110,210,235)">█</span><span style="color:rgb(190,245,255)">██████</span><span style="color:rgb(110,210,235)">█</span><span style="color:rgb(18,80,110)">██</span>
    <span style="color:rgb(18,80,110)">█</span><span style="color:rgb(110,210,235)">█</span><span style="color:rgb(55,170,210)">████████</span><span style="color:rgb(110,210,235)">█</span><span style="color:rgb(18,80,110)">█</span>
    <span style="color:rgb(18,80,110)">█</span><span style="color:rgb(28,120,160)">█</span><span style="color:rgb(55,170,210)">████████</span><span style="color:rgb(28,120,160)">█</span><span style="color:rgb(18,80,110)">█</span>
     <span style="color:rgb(18,80,110)">█</span><span style="color:rgb(28,120,160)">█</span><span style="color:rgb(55,170,210)">██████</span><span style="color:rgb(28,120,160)">█</span><span style="color:rgb(18,80,110)">█</span> 
     <span style="color:rgb(18,80,110)">█</span><span style="color:rgb(28,120,160)">██</span><span style="color:rgb(55,170,210)">████</span><span style="color:rgb(28,120,160)">██</span><span style="color:rgb(18,80,110)">█</span> 
      <span style="color:rgb(18,80,110)">█</span><span style="color:rgb(28,120,160)">██</span><span style="color:rgb(55,170,210)">██</span><span style="color:rgb(28,120,160)">██</span><span style="color:rgb(18,80,110)">█</span>  
      <span style="color:rgb(18,80,110)">██</span><span style="color:rgb(28,120,160)">████</span><span style="color:rgb(18,80,110)">██</span>  
       <span style="color:rgb(18,80,110)">██</span><span style="color:rgb(28,120,160)">██</span><span style="color:rgb(18,80,110)">██</span>   
        <span style="color:rgb(18,80,110)">█</span><span style="color:rgb(28,120,160)">██</span><span style="color:rgb(18,80,110)">█</span>    
         <span style="color:rgb(18,80,110)">██</span>     </pre></span></div></div>""",
        ),
        'building': (
            'Act 2 · Building worlds',
            r"""<style>@keyframes building_moBlink{0%,93%{opacity:0}94%,96%{opacity:1}97%,100%{opacity:0}}@keyframes building_moBuild{0%,100%{transform:translateY(0) scaleX(1) scaleY(1)}4%{transform:translateY(0) scaleX(1.05) scaleY(0.93)}7%{transform:translateY(-2px) scaleX(0.97) scaleY(1.05)}12%{transform:translateY(0) scaleX(1) scaleY(1)}24%{transform:translateY(0) scaleX(1.05) scaleY(0.93)}27%{transform:translateY(-2px) scaleX(0.97) scaleY(1.05)}32%{transform:translateY(0) scaleX(1) scaleY(1)}44%{transform:translateY(0) scaleX(1.05) scaleY(0.93)}47%{transform:translateY(-2px) scaleX(0.97) scaleY(1.05)}52%{transform:translateY(0) scaleX(1) scaleY(1)}}@keyframes building_drop0{0%,8%{opacity:0;transform:translateY(-26px)}13%{opacity:1;transform:translateY(6px)}17%{transform:translateY(0)}93%{opacity:1;transform:translateY(0)}100%{opacity:0}}@keyframes building_drop1{0%,28%{opacity:0;transform:translateY(-26px)}33%{opacity:1;transform:translateY(6px)}37%{transform:translateY(0)}93%{opacity:1;transform:translateY(0)}100%{opacity:0}}@keyframes building_drop2{0%,48%{opacity:0;transform:translateY(-26px)}53%{opacity:1;transform:translateY(6px)}57%{transform:translateY(0)}93%{opacity:1;transform:translateY(0)}100%{opacity:0}}</style><div style="display:flex;align-items:center;justify-content:center;gap:54px;width:100%;box-sizing:border-box;background:transparent;padding:20px 16px;border-radius:10px;overflow:hidden"><span style="display:inline-block;transform-origin:bottom center;animation:building_moBuild 4.6s ease-in-out infinite"><span style="position:relative;display:inline-block;line-height:0"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">             <span style="color:rgb(124,140,66)">█</span><span style="color:rgb(155,172,85)">█</span>  
                <span style="color:rgb(124,140,64)">█</span><span style="color:rgb(95,110,51)">█</span> <span style="color:rgb(155,172,85)">█</span><span style="color:rgb(155,172,86)">█</span>
             <span style="color:rgb(155,172,87)">█</span>   <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(157,171,85)">█</span><span style="color:rgb(155,172,85)">█</span> 
        <span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span>    
      <span style="color:rgb(94,111,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,66)">█</span>   
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(155,172,85)">███</span><span style="color:rgb(124,140,64)">█</span>  
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">█████████</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,64)">██</span> 
    <span style="color:rgb(95,111,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span> 
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">█████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,50)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">█████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,65)">█</span><span style="color:rgb(124,140,64)">███</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(95,110,48)">█</span> 
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">██</span> 
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">█</span>  
      <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(124,140,64)">█</span>  <span style="color:rgb(124,140,64)">███</span>  <span style="color:rgb(124,140,64)">██</span>    </pre><span style="position:absolute;top:0;left:0;animation:building_moBlink 4.3s steps(1,end) infinite"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">             <span style="color:rgb(124,140,66)">█</span><span style="color:rgb(155,172,85)">█</span>  
                <span style="color:rgb(124,140,64)">█</span><span style="color:rgb(95,110,51)">█</span> <span style="color:rgb(155,172,85)">█</span><span style="color:rgb(155,172,86)">█</span>
             <span style="color:rgb(155,172,87)">█</span>   <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(157,171,85)">█</span><span style="color:rgb(155,172,85)">█</span> 
        <span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span>    
      <span style="color:rgb(94,111,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,66)">█</span>   
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(155,172,85)">███</span><span style="color:rgb(124,140,64)">█</span>  
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">█████████</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,64)">██</span> 
    <span style="color:rgb(95,111,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span> 
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(95,110,50)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">█████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,65)">█</span><span style="color:rgb(124,140,64)">███</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(95,110,48)">█</span> 
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">██</span> 
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">█</span>  
      <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(124,140,64)">█</span>  <span style="color:rgb(124,140,64)">███</span>  <span style="color:rgb(124,140,64)">██</span>    </pre></span></span></span><div style="display:grid;grid-template-columns:repeat(5,30px);grid-auto-rows:30px;gap:1px"><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;opacity:.30"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;animation:building_drop0 4.6s ease-in infinite"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███</span><span style="color:rgb(240,60,140)">█</span><span style="color:rgb(78,150,74)">███</span>
    <span style="color:rgb(78,150,74)">██</span><span style="color:rgb(240,60,140)">█</span><span style="color:rgb(255,150,196)">█</span><span style="color:rgb(240,60,140)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(240,60,140)">█████</span><span style="color:rgb(78,150,74)">█</span>
    <span style="color:rgb(78,150,74)">██</span><span style="color:rgb(240,60,140)">███</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███</span><span style="color:rgb(240,60,140)">█</span><span style="color:rgb(78,150,74)">███</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;opacity:.30"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;animation:building_drop1 4.6s ease-in infinite"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(40,54,108)">███████</span>
    <span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">███</span><span style="color:rgb(40,54,108)">███</span>
    <span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">█</span><span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">█</span><span style="color:rgb(40,54,108)">███</span>
    <span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">███</span><span style="color:rgb(40,54,108)">███</span>
    <span style="color:rgb(40,54,108)">███████</span>
    <span style="color:rgb(40,54,108)">███████</span>
    <span style="color:rgb(26,36,76)">███████</span></pre></div></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(46,106,196)">█</span><span style="color:rgb(72,142,228)">█</span><span style="color:rgb(46,106,196)">█</span><span style="color:rgb(72,142,228)">█</span><span style="color:rgb(46,106,196)">█</span><span style="color:rgb(72,142,228)">█</span><span style="color:rgb(46,106,196)">█</span>
    <span style="color:rgb(72,142,228)">███████</span>
    <span style="color:rgb(46,106,196)">███████</span>
    <span style="color:rgb(72,142,228)">███████</span>
    <span style="color:rgb(46,106,196)">███████</span>
    <span style="color:rgb(72,142,228)">███████</span>
    <span style="color:rgb(30,76,150)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(46,106,196)">█</span><span style="color:rgb(72,142,228)">█</span><span style="color:rgb(46,106,196)">█</span><span style="color:rgb(72,142,228)">█</span><span style="color:rgb(46,106,196)">█</span><span style="color:rgb(72,142,228)">█</span><span style="color:rgb(46,106,196)">█</span>
    <span style="color:rgb(72,142,228)">███████</span>
    <span style="color:rgb(46,106,196)">███████</span>
    <span style="color:rgb(72,142,228)">███████</span>
    <span style="color:rgb(46,106,196)">███████</span>
    <span style="color:rgb(72,142,228)">███████</span>
    <span style="color:rgb(30,76,150)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">██</span><span style="color:rgb(240,205,70)">███</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">██</span><span style="color:rgb(240,205,70)">█</span><span style="color:rgb(78,150,74)">█</span><span style="color:rgb(240,205,70)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">██</span><span style="color:rgb(240,205,70)">███</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███</span><span style="color:rgb(240,205,70)">█</span><span style="color:rgb(78,150,74)">███</span>
    <span style="color:rgb(78,150,74)">███</span><span style="color:rgb(240,205,70)">██</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;opacity:.30"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;animation:building_drop2 4.6s ease-in infinite"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(255,192,72)">█</span><span style="color:rgb(235,112,40)">█</span><span style="color:rgb(255,192,72)">█</span><span style="color:rgb(235,112,40)">█</span><span style="color:rgb(255,192,72)">█</span><span style="color:rgb(235,112,40)">█</span><span style="color:rgb(255,192,72)">█</span>
    <span style="color:rgb(235,112,40)">███████</span>
    <span style="color:rgb(235,112,40)">██</span><span style="color:rgb(255,206,92)">█</span><span style="color:rgb(235,112,40)">████</span>
    <span style="color:rgb(235,112,40)">█████</span><span style="color:rgb(255,206,92)">█</span><span style="color:rgb(235,112,40)">█</span>
    <span style="color:rgb(235,112,40)">███████</span>
    <span style="color:rgb(235,112,40)">███████</span>
    <span style="color:rgb(150,56,20)">███████</span></pre></div></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(120,90,32)">███████</span>
    <span style="color:rgb(120,90,32)">█</span><span style="color:rgb(225,182,72)">█████</span><span style="color:rgb(120,90,32)">█</span>
    <span style="color:rgb(120,90,32)">█</span><span style="color:rgb(225,182,72)">█████</span><span style="color:rgb(120,90,32)">█</span>
    <span style="color:rgb(120,90,32)">█</span><span style="color:rgb(225,182,72)">██</span><span style="color:rgb(60,44,16)">█</span><span style="color:rgb(225,182,72)">██</span><span style="color:rgb(120,90,32)">█</span>
    <span style="color:rgb(120,90,32)">█</span><span style="color:rgb(225,182,72)">█████</span><span style="color:rgb(120,90,32)">█</span>
    <span style="color:rgb(120,90,32)">█</span><span style="color:rgb(225,182,72)">█████</span><span style="color:rgb(120,90,32)">█</span>
    <span style="color:rgb(120,90,32)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(40,54,108)">███████</span>
    <span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">███</span><span style="color:rgb(40,54,108)">███</span>
    <span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">█</span><span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">█</span><span style="color:rgb(40,54,108)">███</span>
    <span style="color:rgb(40,54,108)">█</span><span style="color:rgb(74,96,170)">███</span><span style="color:rgb(40,54,108)">███</span>
    <span style="color:rgb(40,54,108)">███████</span>
    <span style="color:rgb(40,54,108)">███████</span>
    <span style="color:rgb(26,36,76)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">█████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">████</span><span style="color:rgb(100,178,96)">█</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div><div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center"><pre style="font:6px/0.6 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(78,150,74)">███████</span>
    <span style="color:rgb(78,150,74)">██</span><span style="color:rgb(232,206,112)">███</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(232,206,112)">██</span><span style="color:rgb(78,150,74)">█</span><span style="color:rgb(232,206,112)">██</span><span style="color:rgb(78,150,74)">█</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(232,206,112)">█</span><span style="color:rgb(78,150,74)">███</span><span style="color:rgb(232,206,112)">█</span><span style="color:rgb(78,150,74)">█</span>
    <span style="color:rgb(78,150,74)">█</span><span style="color:rgb(232,206,112)">██</span><span style="color:rgb(78,150,74)">█</span><span style="color:rgb(232,206,112)">██</span><span style="color:rgb(78,150,74)">█</span>
    <span style="color:rgb(78,150,74)">██</span><span style="color:rgb(232,206,112)">███</span><span style="color:rgb(78,150,74)">██</span>
    <span style="color:rgb(54,116,52)">███████</span></pre></div></div></div>""",
        ),
        'twist': (
            'The Twist · a tiny oracle already solved it',
            r"""<style>@keyframes twist_moBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}@keyframes twist_fill0{0%,4.0%{opacity:0}6.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill1{0%,6.0%{opacity:0}8.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill2{0%,8.0%{opacity:0}10.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill3{0%,10.0%{opacity:0}12.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill4{0%,12.0%{opacity:0}14.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill5{0%,14.0%{opacity:0}16.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill6{0%,16.0%{opacity:0}18.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill7{0%,18.0%{opacity:0}20.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill8{0%,20.0%{opacity:0}22.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill9{0%,22.0%{opacity:0}24.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill10{0%,24.0%{opacity:0}26.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill11{0%,26.0%{opacity:0}28.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill12{0%,28.0%{opacity:0}30.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill13{0%,30.0%{opacity:0}32.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill14{0%,32.0%{opacity:0}34.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill15{0%,34.0%{opacity:0}36.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill16{0%,36.0%{opacity:0}38.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill17{0%,38.0%{opacity:0}40.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill18{0%,40.0%{opacity:0}42.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill19{0%,42.0%{opacity:0}44.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill20{0%,44.0%{opacity:0}46.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill21{0%,46.0%{opacity:0}48.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill22{0%,48.0%{opacity:0}50.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill23{0%,50.0%{opacity:0}52.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill24{0%,52.0%{opacity:0}54.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill25{0%,54.0%{opacity:0}56.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill26{0%,56.0%{opacity:0}58.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill27{0%,58.0%{opacity:0}60.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_fill28{0%,60.0%{opacity:0}62.0%,90%{opacity:1}100%{opacity:0}}@keyframes twist_moSurprise{0%,60%{opacity:0}64%,88%{opacity:1}92%,100%{opacity:0}}@keyframes twist_exPop{0%,60%{opacity:0;transform:scale(.4)}66%{opacity:1;transform:scale(1.2)}70%,88%{opacity:1;transform:scale(1)}92%,100%{opacity:0;transform:scale(1)}}</style><div style="display:flex;align-items:center;justify-content:center;gap:50px;width:100%;box-sizing:border-box;background:transparent;padding:34px 16px;border-radius:10px;overflow:hidden"><span style="position:relative;display:inline-block;line-height:0"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">                          <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                              <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">████████████████████████████</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        </pre><span style="position:absolute;top:0;left:0;animation:twist_moSurprise 4.0s steps(1,end) infinite"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">                          <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                              <span style="color:rgb(124,140,66)">██</span><span style="color:rgb(155,172,85)">██</span>    
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                            <span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,51)">██</span>  <span style="color:rgb(155,172,85)">██</span><span style="color:rgb(155,172,86)">██</span>
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
                      <span style="color:rgb(155,172,87)">██</span>      <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(157,171,85)">██</span><span style="color:rgb(155,172,85)">██</span>  
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
            <span style="color:rgb(124,140,64)">████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span>        
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
        <span style="color:rgb(94,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(155,172,84)">██</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,66)">██</span>      
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">████████████████</span><span style="color:rgb(155,172,85)">██████</span><span style="color:rgb(124,140,64)">██</span>    
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████</span><span style="color:rgb(155,172,85)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,111,49)">██</span><span style="color:rgb(124,140,64)">██████████████████████████████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span>  
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(95,110,50)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,49)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,65)">██</span><span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(94,111,48)">██</span>
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(124,140,64)">██████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,48)">██</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
    <span style="color:rgb(95,110,51)">██</span><span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">████</span>  
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
      <span style="color:rgb(95,110,48)">██</span><span style="color:rgb(124,140,63)">██</span><span style="color:rgb(124,140,64)">██████████████████████</span><span style="color:rgb(95,110,48)">██</span>    
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        
        <span style="color:rgb(94,111,48)">██</span><span style="color:rgb(124,140,64)">██</span>    <span style="color:rgb(124,140,64)">██████</span>    <span style="color:rgb(124,140,64)">████</span>        </pre></span><span style="position:absolute;top:-14px;left:-10px;transform-origin:bottom right;animation:twist_exPop 4.0s ease-out infinite"><pre style="font:11px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"><span style="color:rgb(255,242,170)">██</span>
    <span style="color:rgb(255,242,170)">██</span>
    <span style="color:rgb(255,242,170)">██</span>
    <span style="color:rgb(245,205,70)">██</span>
    <span style="color:rgb(245,205,70)">██</span>

    <span style="color:rgb(210,165,40)">██</span></pre></span></span><div style="padding:0 30px"><div style="transform:rotateX(56deg) rotateZ(-45deg) scale(1.15);transform-origin:center center"><pre style="font:13px/0.78 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0"><span style="color:rgb(255,225,90);animation:twist_fill0 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(255,225,90);animation:twist_fill6 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill7 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill8 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span>
    <span style="color:rgb(255,225,90);animation:twist_fill1 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(255,225,90);animation:twist_fill5 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(255,225,90);animation:twist_fill9 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span>
    <span style="color:rgb(255,225,90);animation:twist_fill2 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill3 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill4 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(255,225,90);animation:twist_fill10 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span>
    <span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(255,225,90);animation:twist_fill11 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span>
    <span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(255,225,90);animation:twist_fill14 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill13 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill12 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill21 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill22 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill23 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span>
    <span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(255,225,90);animation:twist_fill15 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(255,225,90);animation:twist_fill20 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(255,225,90);animation:twist_fill24 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span>
    <span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(255,225,90);animation:twist_fill16 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill17 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill18 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill19 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(255,225,90);animation:twist_fill25 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(140,144,156)">▣</span>
    <span style="color:rgb(44,48,57)">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(255,225,90);animation:twist_fill26 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill27 4.0s linear infinite">█</span><span style="color:rgb(255,225,90);animation:twist_fill28 4.0s linear infinite">█</span></pre></div></div><span style="display:inline-block;animation:twist_moBob 2.6s ease-in-out 0s infinite;margin-top:0px"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">        <span style="color:rgb(120,225,245)">████</span>        
            <span style="color:rgb(120,225,245)">████</span>        
              <span style="color:rgb(120,225,245)">██</span>        
              <span style="color:rgb(120,225,245)">██</span>        
        <span style="color:rgb(95,115,130)">████████████</span>    
        <span style="color:rgb(95,115,130)">████████████</span>    
      <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">████████████</span><span style="color:rgb(95,115,130)">██</span>  
      <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">████████████</span><span style="color:rgb(95,115,130)">██</span>  
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">██</span>  <span style="color:rgb(45,225,255)">██</span>    <span style="color:rgb(45,225,255)">██</span>  <span style="color:rgb(150,175,190)">██</span><span style="color:rgb(95,115,130)">██</span>
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">██</span>  <span style="color:rgb(45,225,255)">██</span>    <span style="color:rgb(45,225,255)">██</span>  <span style="color:rgb(150,175,190)">██</span><span style="color:rgb(95,115,130)">██</span>
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">██</span>            <span style="color:rgb(150,175,190)">██</span><span style="color:rgb(95,115,130)">██</span>
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">██</span>            <span style="color:rgb(150,175,190)">██</span><span style="color:rgb(95,115,130)">██</span>
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">██</span>  <span style="color:rgb(70,200,235)">████████</span>  <span style="color:rgb(150,175,190)">██</span><span style="color:rgb(95,115,130)">██</span>
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">██</span>  <span style="color:rgb(70,200,235)">████████</span>  <span style="color:rgb(150,175,190)">██</span><span style="color:rgb(95,115,130)">██</span>
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">████████████████</span><span style="color:rgb(95,115,130)">██</span>
    <span style="color:rgb(95,115,130)">██</span><span style="color:rgb(150,175,190)">████████████████</span><span style="color:rgb(95,115,130)">██</span>
      <span style="color:rgb(95,115,130)">████████████████</span>  
      <span style="color:rgb(95,115,130)">████████████████</span>  
      <span style="color:rgb(95,115,130)">██</span>  <span style="color:rgb(95,115,130)">████████</span>  <span style="color:rgb(95,115,130)">██</span>  
      <span style="color:rgb(95,115,130)">██</span>  <span style="color:rgb(95,115,130)">████████</span>  <span style="color:rgb(95,115,130)">██</span>  
      <span style="color:rgb(95,115,130)">████</span>  <span style="color:rgb(95,115,130)">████</span>  <span style="color:rgb(95,115,130)">████</span>  
      <span style="color:rgb(95,115,130)">████</span>  <span style="color:rgb(95,115,130)">████</span>  <span style="color:rgb(95,115,130)">████</span>  
      <span style="color:rgb(95,115,130)">██</span>            <span style="color:rgb(95,115,130)">██</span>  
      <span style="color:rgb(95,115,130)">██</span>            <span style="color:rgb(95,115,130)">██</span>  
    <span style="color:rgb(95,115,130)">████</span>            <span style="color:rgb(95,115,130)">████</span>
    <span style="color:rgb(95,115,130)">████</span>            <span style="color:rgb(95,115,130)">████</span></pre></span></div>""",
        ),
        'ask_model': (
            'Act 3 · Ask the model',
            r"""<style>@keyframes ask_model_moBlink{0%,93%{opacity:0}94%,96%{opacity:1}97%,100%{opacity:0}}@keyframes ask_model_bossThink{0%,100%{transform:scaleX(1) scaleY(1)}28%{transform:scaleX(1.03) scaleY(0.91)}54%{transform:scaleX(0.975) scaleY(1.10)}78%{transform:scaleX(1.015) scaleY(0.97)}}@keyframes ask_model_bossGlow{0%,100%{opacity:.8}50%{opacity:1}}</style><div style="display:flex;align-items:center;justify-content:center;gap:90px;width:100%;box-sizing:border-box;background:transparent;padding:22px 16px;border-radius:10px;overflow:hidden"><div style="display:flex;align-items:flex-end"><span style="position:relative;display:inline-block;line-height:0"><pre style="font:6px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">             <span style="color:rgb(124,140,66)">█</span><span style="color:rgb(155,172,85)">█</span>  
                <span style="color:rgb(124,140,64)">█</span><span style="color:rgb(95,110,51)">█</span> <span style="color:rgb(155,172,85)">█</span><span style="color:rgb(155,172,86)">█</span>
             <span style="color:rgb(155,172,87)">█</span>   <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(157,171,85)">█</span><span style="color:rgb(155,172,85)">█</span> 
        <span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span>    
      <span style="color:rgb(94,111,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,66)">█</span>   
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(155,172,85)">███</span><span style="color:rgb(124,140,64)">█</span>  
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">█████████</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,64)">██</span> 
    <span style="color:rgb(95,111,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span> 
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">█████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(95,110,50)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">█████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,65)">█</span><span style="color:rgb(124,140,64)">███</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(95,110,48)">█</span> 
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">██</span> 
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">█</span>  
      <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(124,140,64)">█</span>  <span style="color:rgb(124,140,64)">███</span>  <span style="color:rgb(124,140,64)">██</span>    </pre><span style="position:absolute;top:0;left:0;animation:ask_model_moBlink 4.3s steps(1,end) infinite"><pre style="font:6px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">             <span style="color:rgb(124,140,66)">█</span><span style="color:rgb(155,172,85)">█</span>  
                <span style="color:rgb(124,140,64)">█</span><span style="color:rgb(95,110,51)">█</span> <span style="color:rgb(155,172,85)">█</span><span style="color:rgb(155,172,86)">█</span>
             <span style="color:rgb(155,172,87)">█</span>   <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(157,171,85)">█</span><span style="color:rgb(155,172,85)">█</span> 
        <span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span>    
      <span style="color:rgb(94,111,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,66)">█</span>   
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(155,172,85)">███</span><span style="color:rgb(124,140,64)">█</span>  
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">█████████</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,64)">██</span> 
    <span style="color:rgb(95,111,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span> 
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(95,110,50)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">█████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,65)">█</span><span style="color:rgb(124,140,64)">███</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(124,140,64)">██████████████</span><span style="color:rgb(95,110,48)">█</span> 
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">██</span> 
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">███████████</span><span style="color:rgb(95,110,48)">█</span>  
      <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(124,140,64)">█</span>  <span style="color:rgb(124,140,64)">███</span>  <span style="color:rgb(124,140,64)">██</span>    </pre></span></span></div><div style="display:flex;flex-direction:column;align-items:center;line-height:0"><span style="display:inline-block;transform-origin:bottom center;animation:ask_model_bossThink 1.5s ease-in-out infinite"><span style="display:inline-block;animation:ask_model_bossGlow 1.5s ease-in-out infinite"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">              <span style="color:rgb(0,0,0)">█████</span>               
            <span style="color:rgb(0,0,0)">██████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██████</span>         
         <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(254,150,223)">█████████████████</span><span style="color:rgb(0,0,0)">███</span>      
       <span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██</span>    
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(197,134,253)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">███████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███████████████████</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██████████</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█████</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  </pre></span></span><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(162,168,249)">█</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(162,168,249)">██</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(162,168,249)">██</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span>
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">██</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>  
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">██████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">████████</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█████████████████████</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█████████████████</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>    
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█████████████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  <span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">██████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">█</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span>    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">█████████████</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>     <span style="color:rgb(0,0,0)">█████████████</span>   <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span>   
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span>       <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">█</span><span style="color:rgb(195,195,195)">█████</span><span style="color:rgb(164,164,164)">█</span><span style="color:rgb(0,0,0)">█</span>     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">██</span>     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">██</span><span style="color:rgb(195,195,195)">███</span><span style="color:rgb(164,164,164)">██</span><span style="color:rgb(0,0,0)">█</span>     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">█</span>   <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">█</span>  <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(107,107,107)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">███████</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(107,107,107)">█</span><span style="color:rgb(0,0,0)">█</span>  <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>    
       <span style="color:rgb(0,0,0)">████</span>  <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(107,107,107)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">█</span>  <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">█</span>  <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(107,107,107)">█</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>     
             <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">█</span>   <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">█</span>   <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>      
             <span style="color:rgb(0,0,0)">███</span>   <span style="color:rgb(0,0,0)">███</span>   <span style="color:rgb(0,0,0)">███</span> <span style="color:rgb(0,0,0)">██</span>       
            <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">█</span><span style="color:rgb(195,195,195)">██</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(164,164,164)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(195,195,195)">██</span><span style="color:rgb(164,164,164)">█</span><span style="color:rgb(0,0,0)">█</span>         
            <span style="color:rgb(0,0,0)">█████</span> <span style="color:rgb(0,0,0)">█████</span> <span style="color:rgb(0,0,0)">█████</span>         
            <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(107,107,107)">██</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(64,64,64)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(107,107,107)">██</span><span style="color:rgb(64,64,64)">█</span><span style="color:rgb(0,0,0)">█</span>         
             <span style="color:rgb(0,0,0)">███</span>   <span style="color:rgb(0,0,0)">███</span>   <span style="color:rgb(0,0,0)">███</span>          </pre></div></div>""",
        ),
    }


    def banner_html(name):
        """Raw self-contained HTML string for a banner."""
        return BANNERS[name][1]


    def banner(name, title=True):
        """Return a marimo renderable for a banner (heading + visual, or just visual)."""
        import marimo as mo
        label, html = BANNERS[name]
        vis = mo.Html(html)
        return mo.vstack([mo.md(f'**{label}**'), vis]) if title else vis


    # featherlm — our LLM loader/parser/optimizer layer, (accelerate + kernels are in the script header). One-time on a fresh kernel; cached.
    import sys as _sys_fl, importlib.util as _ilu_fl
    for _pkg, _spec in (("accelerate", "accelerate"),
                        ("featherlm", "git+https://github.com/orbrx/featherlm.git")):
        if _ilu_fl.find_spec(_pkg) is None:
            subprocess.run([_sys_fl.executable, "-m", "pip", "install", "-q", _spec])
    # transformers (imported above, before accelerate) cached it as unavailable; clear that
    # so featherlm's device_map works without a kernel restart.
    try:
        from transformers.utils import is_accelerate_available as _iaa_fl
        _iaa_fl.cache_clear()
    except Exception:
        pass
    import featherlm


@app.cell(hide_code=True)
def intro():
    mo.vstack([
        mo.md('# Can LLMs Plan?'),
        banner('title', title=False),
        paper_card,
        mo.md(r'''
    An exploration of the planning abilities of modern LLMs, inspired by the
    LLM-BabyBench paper. Hand a frontier model a tiny BabyAI/MiniGrid world
    *as text* and it can often tell you what a single action does. Ask it
    to **plan the whole sequence** to a goal, though, and it falls apart, on a
    world a child solves in seconds. LLM-BabyBench pins this gap down with three
    tasks: Predict (the consequence of an action), Plan (a full low-level
    action sequence), and Decompose (break a goal into subgoals). It finds
    that grounded, multi-step planning is exactly where today's models break.

    This notebook turns that finding into something you can play with: guide Mo
    the Mossball through a world by hand, generate harder ones to stress a planner,
    then hand the *very same* worlds to a live model and watch where it gets lost.

    The grid worlds are live [anywidget](https://anywidget.dev) components you can
    drag, edit tile by tile, and re-roll, all in the browser. And the model runs
    for real on the molab GPU, not a canned transcript: every attempt you see
    below is generated live, mistakes and all.
    '''),
    ])
    return


@app.function(hide_code=True)
def paper_note(body_html):
    """Recurring honesty aside — our '📘 vs. the paper' fidelity motif (the red line)."""
    return mo.Html(f"""
<div style="border-left:3px solid rgba(78,121,180,0.6);background:rgba(78,121,180,0.06);
            border-radius:0 8px 8px 0;padding:11px 16px;margin:12px 0;font-size:0.9rem;line-height:1.55;">
  <div style="font-weight:800;opacity:0.6;font-size:0.7rem;letter-spacing:0.09em;
              text-transform:uppercase;margin-bottom:5px;">📘 vs. the paper</div>
  <div style="opacity:0.92;">{body_html}</div>
</div>""")


@app.cell(hide_code=True)
def fidelity_anchor():
    paper_note(
        "We <b>own our differences</b>. This notebook isn't a reproduction of "
        "<i>LLM-BabyBench</i>. We built Wanderland first, then chose the paper whose "
        "question it embodies. The platform is faithful BabyAI/MiniGrid: an agent with a "
        "facing, turn/forward locomotion, colored objects, doors you unlock with a matching key. "
        "But the game is ours: instead of the paper's formal command language, Mo plays "
        "something closer to Pac-Man: eat every gem, dodge the lava, reach the exit. "
        "Little 📘 notes like this flag exactly what we keep and what we trade, as we go."
    )
    return


@app.cell(hide_code=True)
def act1_intro():
    mo.vstack([
        mo.md('## Act 1 · Meet Mo the Mossball'),
        banner('meet_mo', title=False),
        mo.md(r'''
    This is **Mo the Mossball** (the unofficial [marimo](https://marimo.io) mascot),
    and his home is [Wanderland](https://github.com/ktaletsk/wanderland), an
    open-source low-poly 3D coding playground I built as an
    [anywidget](https://anywidget.dev) for Python notebooks (`pip install
    wanderland`). You write Python to steer him through a world, exactly the kind
    of grounded, grid-world planning LLM-BabyBench puts language models through.
    *(Curious how Mo came to be? Read [the story behind Wanderland](https://taletskiy.com/blogs/wanderland/?utm_source=molab&utm_medium=notebook&utm_campaign=wanderland-launch&utm_content=act1-meet-mo).)*

    **Can you guide him to find the gems and clear the level?** Try to think
    about the plan *in your head* before running the code. Use the provided
    functions to write the solution.

    Mo starts in the top-left corner, facing east (to the right).
    Walk him *onto* a gem's tile, then `collect_gem()` to pick it up. Grab
    both gems, then land on the golden ring to win. The pond in the middle is
    impassable, so go around.

    The commands this world allows (its action space):

    - `move_forward()`
    - `turn_left()` / `turn_right()`
    - `collect_gem()`: collect the gem on the tile Mo is standing on
    '''),
        mo.callout(
            mo.md(r'''
    **Write a plan for Mo.** Type the commands or tap helpful buttons to append commands for you,
    one command per line, in the code editor below. Your edits load into the
    scene automatically thanks to Marimo's reactive model; then press &#9654; Run My Code to watch Mo go.
    '''),
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def fidelity_actions():
    paper_note(
        "BabyAI's action space is six verbs: <code>forward, left, right, pickup, drop, open</code>. "
        "Mo's locomotion is identical, but his interaction verb here is <code>collect_gem()</code>: a "
        "gentle <i>walk-onto-the-tile-and-grab</i> variant, not BabyAI's <code>pickup</code> of the "
        "object you <i>face</i>. The Wanderland library implements the faithful "
        "<code>pickup/drop/toggle</code> (with solid, blocking objects) too; this notebook just leads "
        "with the friendlier version."
    )
    return


@app.cell(hide_code=True)
def act1_code_state():
    # Single source of truth for Mo's program. The command buttons append to it and
    # the editor edits it -- both flow through this state so they never disagree.
    get_code, set_code = mo.state(
        "move_forward()\n"
        "move_forward()\n"
        "move_forward()\n"
        "collect_gem()\n"
    )
    return get_code, set_code


@app.cell(hide_code=True)
def act1_solution_editor(get_code, set_code):
    # Editor definition only -- it is laid out (with the palette and the 3D scene)
    # in the layout cell below. Kept in its OWN cell so a palette-button tap
    # refreshes it: marimo preserves UI values only within the cell that *defines*
    # the element that triggered the rerun.
    solution_code = mo.ui.code_editor(
        value=get_code(),
        language="python",
        debounce=True,
        on_change=set_code,
        min_height=150,
        show_copy_button=False,
    )
    return (solution_code,)


@app.cell(hide_code=True)
def act1_command_buttons(get_code, set_code):
    # Command palette (rendered in the layout cell). Separate cell from the editor
    # so taps refresh the editor. Buttons are global and bump their value on click
    # so on_change fires.
    def _appender(snippet):
        def _cb(_v):
            cur = get_code().rstrip("\n")
            set_code((cur + "\n" + snippet if cur else snippet) + "\n")
        return _cb


    def _undo_cb(_v):
        lines = get_code().rstrip("\n").split("\n")
        set_code(("\n".join(lines[:-1]) + "\n") if len(lines) > 1 else "")


    _bump = lambda v: (v or 0) + 1
    btn_forward = mo.ui.button(value=0, on_click=_bump, on_change=_appender("move_forward()"), label="move_forward()")
    btn_left = mo.ui.button(value=0, on_click=_bump, on_change=_appender("turn_left()"), label="turn_left()")
    btn_right = mo.ui.button(value=0, on_click=_bump, on_change=_appender("turn_right()"), label="turn_right()")
    btn_gem = mo.ui.button(value=0, on_click=_bump, on_change=_appender("collect_gem()"), label="collect_gem()")
    btn_undo = mo.ui.button(value=0, on_click=_bump, on_change=_undo_cb, label="\u232b undo")
    btn_clear = mo.ui.button(value=0, on_click=_bump, on_change=lambda _v: set_code(""), kind="danger", label="\u2715 clear")
    command_palette = mo.hstack(
        [btn_forward, btn_left, btn_right, btn_gem, btn_undo, btn_clear],
        justify="start", gap=0.4, wrap=True,
    )
    return (command_palette,)


@app.cell(hide_code=True)
def act1_scene():
    # The grid world Mo lives in (the "Gem Path" puzzle), with its own Run button.
    # Defined here; laid out beside the editor in the layout cell.
    world = mo.ui.anywidget(bp.World(bp.puzzles.gem_path()))
    return (world,)


@app.cell(hide_code=True)
def act1_layout(command_palette, solution_code, world):
    # Act 1 playground: editor + command palette on the LEFT, Mo's 3D world on the
    # RIGHT. Tapping a command or typing loads into the scene automatically; press
    # Run My Code in the scene to watch Mo go.
    mo.hstack(
        [
            mo.vstack([
                solution_code,
                command_palette,
            ]),
            world,
        ],
        align="start", gap=1.5, widths=[0.4, 0.6],
    )
    return


@app.cell(hide_code=True)
def act1_load(solution_code, world):
    # Build Mo's program from the editable block above and hand the timeline to
    # the scene (reactive: recomputes whenever you edit the code). Mo only moves
    # when you press the Run button in the scene.
    _ns = {"move_forward": move_forward, "turn_left": turn_left,
           "turn_right": turn_right, "collect_gem": collect_gem}

    def solution():
        exec(compile(solution_code.value, "<solution>", "exec"), _ns)

    try:
        loaded = world.load(solution)
        solution_error = None
    except Exception as _e:
        loaded = None
        solution_error = _e
    return loaded, solution_error


@app.cell(hide_code=True)
def act1_result(loaded, solution_error, world):
    _ = loaded
    _ = world.value
    if solution_error is not None:
        _out = mo.callout(
            mo.md(
                f"""
                Your code didn\'t run. Fix it and Mo will be ready:

                ```
                {type(solution_error).__name__}: {solution_error}
                ```
                """
            ),
            kind="danger",
        )
    elif not world.state.get("finished"):
        _out = mo.callout(
            mo.md("Press the Run button in the scene to send Mo off."),
            kind="info",
        )
    else:
        _won = world.success
        _out = mo.callout(
            mo.md(
                f"""
                Result &nbsp; {"Puzzle solved!" if _won else "Keep trying..."}

                - Gems collected: {world.gems_collected} / {world.total_gems}
                - Reached the goal: {world.reached_goal}
                """
            ),
            kind="success" if _won else "neutral",
        )
    _out
    return


@app.cell(hide_code=True)
def act2_blocks():
    mo.vstack([
        mo.md('## Act 2 · Building & generating worlds'),
        banner('building', title=False),
        mo.md(r'''
    Mo's gem path was about the gentlest world there is. But every world he can
    live in is built from the same small set of pieces:

    **1 · The ground: terrain & obstacles.**
    `.` floor &nbsp; `#` wall &nbsp; `~` water (impassable) &nbsp; `!` lava
    (walkable, but stepping on it ends the run) &nbsp; `O` the goal tile.

    **2 · Objects on a tile.** Gems to collect (`g` walk onto it, `G` pick it
    up), plus colored keys, balls, boxes (which can hide something inside),
    and doors. A *locked* door only opens if Mo carries a key of the matching
    color. The start tile carries Mo's facing: `>` `<` `^` `v`.

    **3 · Actions: the action space.** Each world declares exactly which verbs
    it allows: move (`move_forward`, `turn_left`, `turn_right`) and interact
    (`pickup`, `drop`, `toggle`).

    Put those together and a world is, really, just data. The friendliest
    way to feel that: fill in a grid like a spreadsheet, one token per
    tile, using the glyphs above. Edit any cell below and Mo's world rebuilds
    itself in 3D (and you'll see the ASCII it compiles to).
    '''),
    ])
    return


@app.cell(hide_code=True)
def act2_designer():
    # Your tile palette: . floor · # wall · ~ water · ! lava · O goal · g gem
    # · > start (facing) · Ky yellow key · Ly locked yellow door. One token per
    # tile -- edit cells, add/remove rows, and the world below rebuilds live.
    _starter = [
        [">", ".", ".", "g", "."],
        [".", "#", "~", "~", "."],
        ["Ky", ".", "!", ".", "."],
        [".", "Ly", "#", ".", "O"],
    ]
    grid_df = pd.DataFrame(_starter, columns=[str(i) for i in range(len(_starter[0]))])
    grid_designer = mo.ui.data_editor(grid_df, label="Tile editor")
    return (grid_designer,)


@app.cell(hide_code=True)
def act2_designer_scene(grid_designer):
    def _grid_to_ascii(val):
        rows = val.values.tolist() if hasattr(val, "values") else [list(r.values()) for r in val]
        return "\n".join(" ".join((str(c).strip() or ".") for c in row) for row in rows)

    designed_ascii = _grid_to_ascii(grid_designer.value)
    try:
        designed = bp.from_ascii(
            "Your Design", designed_ascii,
            actions=("move_forward", "turn_left", "turn_right",
                     "pickup", "drop", "toggle", "collect_gem"),
        )
        designed_world = mo.ui.anywidget(bp.World(designed))
        designed_panel = mo.vstack([
            designed_world,
            mo.md(f"...which compiles to this ASCII map (a world is just data):\n\n```\n{designed_ascii}\n```"),
        ])
    except Exception as _e:
        designed_world = None
        designed_panel = mo.callout(
            mo.md(
                f"**Can't build that world yet:** {_e}\n\n"
                "Every world needs exactly **one** start tile (`>` `<` `^` `v`)."
            ),
            kind="warn",
        )
    return (designed_panel,)


@app.cell(hide_code=True)
def act2_layout(designed_panel, grid_designer):
    # Builder playground: the spreadsheet tile editor on the LEFT, the world it
    # compiles to on the RIGHT. Edit any cell and the 3D scene rebuilds live.
    mo.hstack(
        [grid_designer, designed_panel],
        align="start", gap=1.5, widths=[0.4, 0.6],
    )
    return


@app.cell(hide_code=True)
def act2_generate_intro():
    mo.md(r"""
    ### ...but by hand gets tedious. Let's generate.

    Sketching one world is fun; sketching *hundreds* to stress-test a planner is
    not. So let's generate them, with real terrain, not a tidy two-room
    split. Each roll scatters irregular walls, water, and lava, then
    checks the result against the BFS oracle and rerolls anything it can't
    solve. (So the oracle isn't only a baseline; it's our solvability
    guarantee.)

    - **Size:** the grid's *width* x *height*.
    - **Complexity:** obstacle density, how many gems to gather, and an
      optional locked door that walls Mo into a separate room until he finds
      the matching key.
    - **Seed:** rolls a fresh layout. Not difficulty, just variety.

    *Why split size from complexity?* Because in the paper's own
    experiments that distinction is the whole story: pile on obstacles *within*
    a fixed grid and a model's planning barely budges, but **grow the grid and
    accuracy falls off a cliff**. Grid size, far more than clutter, is what
    breaks the planner (LLM-BabyBench, Choukrani et al., 2025).

    Drag the sliders and a brand-new, always-solvable world appears.
    """)
    return


@app.cell(hide_code=True)
def fidelity_objects():
    paper_note(
        "Walls, and locked doors that open with a matching key, are real BabyAI. "
        "Gems, lava, water, and a goal tile are ours: lava and the goal are "
        "MiniGrid-flavoured; the gems and the collect-<i>every</i>-one objective are pure Wanderland. "
        "BabyAI's other carryables, balls and boxes, live in the library but this "
        "notebook doesn't place them."
    )
    return


@app.cell(hide_code=True)
def world_generator():
    COLORS = ("red", "green", "blue", "purple", "yellow", "grey")

    def random_world(seed, cols, rows, gems=1, density=0.18, locked=False, attempts=500):
        """Scatter irregular walls + water + lava, drop gems and a goal, and -- when
        ``locked`` -- split the map into two rooms with a locked door + matching key.
        Every candidate is checked by the BFS oracle and rerolled if unsolvable
        (easing the obstacle density if rolls keep failing)."""
        for a in range(attempts):
            r = random.Random(seed * 100003 + a)
            d = density * (1 - a / attempts)
            actions = ["move_forward", "turn_left", "turn_right"]
            objects, walls, water, lava = {}, [], [], []

            if locked:
                # two rooms split by a wall column with a single door gap
                wc = r.randint(2, cols - 2)
                door_row = r.randint(0, rows - 1)
                walls = [(wc, w) for w in range(rows) if w != door_row]
                color = r.choice(COLORS)
                objects[(wc, door_row)] = bp.Obj("door", color, "locked")
                left  = [(c, w) for c in range(wc)         for w in range(rows) if (c, w) != (wc - 1, door_row)]
                right = [(c, w) for c in range(wc + 1, cols) for w in range(rows) if (c, w) != (wc + 1, door_row)]
                r.shuffle(left); r.shuffle(right)
                start = left.pop()
                objects[left.pop()] = bp.Obj("key", color)   # matching key in Mo's room
                goal = right.pop()
                pool = left + right
                actions += ["pickup", "toggle"]
            else:
                cells = [(c, w) for c in range(cols) for w in range(rows)]
                r.shuffle(cells)
                start, goal = cells.pop(), cells.pop()
                pool = cells

            free = []
            for cell in pool:
                t = r.random()
                if t < d:            walls.append(cell)
                elif t < d * 1.4:    water.append(cell)
                elif t < d * 1.6:    lava.append(cell)
                else:                free.append(cell)
            r.shuffle(free)
            for _ in range(min(gems, len(free))):
                objects[free.pop()] = bp.Obj("gem", blocking=False)
            if any(o.type == "gem" for o in objects.values()):
                actions.append("collect_gem")

            p = bp.Puzzle(
                name="Wildlands", cols=cols, rows=rows, start=start,
                actions=tuple(actions), heading=r.choice("NESW"),
                objects=objects, goal=goal, gaps=water, walls=walls, lava=lava,
            )
            if bp.solve(p) is not None:
                return p
        return None

    return (random_world,)


@app.cell(hide_code=True)
def gen_controls():
    gen_cols = mo.ui.slider(4, 10, value=7, label="width", show_value=True, full_width=True)
    gen_rows = mo.ui.slider(3, 8, value=6, label="height", show_value=True, full_width=True)
    gen_seed = mo.ui.slider(0, 30, value=7, label="seed", show_value=True, full_width=True)
    gen_gems = mo.ui.slider(0, 4, value=2, label="gems", show_value=True, full_width=True)
    gen_density = mo.ui.slider(0.0, 0.35, value=0.2, step=0.01, label="obstacles", show_value=True, full_width=True)
    gen_locked = mo.ui.switch(value=False, label="Doors and locks")

    # Polished control panel: a themed card (matches the paper-reference card),
    # uppercase moss-green section headers, full-width sliders with value readouts.
    _hdr = lambda t: mo.md(
        f"<span style='font-size:0.7rem;font-weight:800;letter-spacing:0.09em;"
        f"text-transform:uppercase;color:rgb(124,140,64)'>{t}</span>"
    )
    _cap = lambda t: mo.md(f"<span style='font-size:0.78rem;opacity:0.6'>{t}</span>")

    gen_controls_panel = mo.vstack([
        mo.md("**\U0001F3B2 Generator controls**"),
        _hdr("Size"),
        gen_cols,
        gen_rows,
        _hdr("Complexity"),
        gen_density,
        gen_gems,
        gen_locked,
        _hdr("Seed"),
        _cap("just variety, not difficulty"),
        gen_seed,
    ], gap=0.55).style({
        "border": "1px solid rgba(124,140,64,0.35)",
        "border-radius": "12px",
        "padding": "18px 20px",
        "background": "rgba(124,140,64,0.06)",
        "box-shadow": "0 8px 24px rgba(0,0,0,0.05)",
    })
    return (
        gen_cols,
        gen_controls_panel,
        gen_density,
        gen_gems,
        gen_locked,
        gen_rows,
        gen_seed,
    )


@app.cell(hide_code=True)
def gen_scene(
    gen_cols,
    gen_density,
    gen_gems,
    gen_locked,
    gen_rows,
    gen_seed,
    random_world,
):
    # Roll a world from the controls: irregular walls, water and lava -- plus an
    # optional locked door that splits it into two rooms. Always solvable;
    # re-rolls whenever you move a slider.
    gen_puzzle = random_world(
        seed=gen_seed.value,
        cols=gen_cols.value,
        rows=gen_rows.value,
        gems=gen_gems.value,
        density=gen_density.value,
        locked=gen_locked.value,
    )
    gen_env = bp.World(gen_puzzle)
    gen_world = mo.ui.anywidget(gen_env)
    return gen_env, gen_puzzle, gen_world


@app.cell(hide_code=True)
def gen_oracle(gen_env, gen_puzzle):
    # A BFS oracle finds a shortest solving plan; load it so Run replays it.
    gen_plan = bp.solve(gen_puzzle)
    if gen_plan:
        gen_env.act(gen_plan)
    return (gen_plan,)


@app.cell(hide_code=True)
def gen_layout(gen_controls_panel, gen_world):
    # Generator playground: controls on the LEFT, the rolled (always-solvable)
    # world on the RIGHT. Move a slider and a fresh, solvable world appears.
    mo.hstack(
        [gen_controls_panel, gen_world],
        align="start", gap=1.5, widths=[0.4, 0.6],
    )
    return


@app.cell(hide_code=True)
def gen_readout(
    gen_cols,
    gen_gems,
    gen_locked,
    gen_plan,
    gen_puzzle,
    gen_rows,
    gen_world,
):
    _ = gen_world.value  # also refresh after a playback finishes
    _terr = f"{len(gen_puzzle.wall_set)} walls, {len(gen_puzzle.gap_set)} water, {len(gen_puzzle.lava_set)} lava"
    _door = " &middot; locked door + key" if gen_locked.value else ""
    if gen_plan:
        _msg = mo.md(
            f"""
            **This world** &nbsp; {gen_cols.value} x {gen_rows.value} &middot; {_terr}{_door} &middot; {gen_gems.value} gem(s)

            Shortest solution: **{len(gen_plan)} steps**, your difficulty meter.
            Grow the grid, crank the obstacles, add gems, or lock the door and watch
            it climb. Press the Run button in the scene to watch the oracle solve it.
            """
        )
        _kind = "info"
    else:
        _msg = mo.md("**No solution found.** Try another seed.")
        _kind = "neutral"
    mo.callout(_msg, kind=_kind)
    return


@app.cell(hide_code=True)
def act2_why_llms():
    mo.vstack([
        mo.md('### Wait, these worlds came *pre-solved*. By what?'),
        banner('twist', title=False),
        mo.md(r'''
    The difficulty meter already knew the answer, because every generated world
    is run through a tiny **breadth-first search (BFS)** oracle (`solve`):

    - It treats Mo's situation as a state: where he stands, which way he
      faces, what he's carrying, and where every object is.
    - From the start it expands states layer by layer, trying each allowed
      verb, and throws away any state it has already seen.
    - Because BFS reaches every state by the *shortest* path first, the instant
      it hits "all gems collected, standing on the goal," that plan is provably
      the fewest possible steps.

    And it's fast, milliseconds, for two reasons. The world is small,
    fully observed, and deterministic: Mo sees the whole map and every
    action's outcome is exact. And the *seen*-set means each state is explored
    once, so the search never spirals.

    This BFS oracle is exactly the role the paper gives its **OmniBot**: an
    optimal solver that grades every answer by *actually running it* in the
    world (execution-based, not "does the text look right"), and sets the
    100%-success bar a language model is measured against (LLM-BabyBench,
    Choukrani et al., 2025).

    Which raises an uncomfortable question:

    > If a few lines of BFS already solve every one of these worlds,
    > optimally, in milliseconds, why would anyone bother handing the job to a
    > language model?

    That question is what the rest of the notebook is about.
    '''),
    ])
    return


@app.cell(hide_code=True)
def act3_intro():
    mo.vstack([
        mo.md('## Act 3 · Now ask a language model'),
        banner('ask_model', title=False),
        mo.md(r'''
    The oracle had it easy: it sees the whole world as exact coordinates and
    brute-forces the shortest path. Now hand a language model that *same*
    structured description, ask it for a plan, and replay its answer in Mo's world.

    Same world, same goal, same action space the oracle just solved. Can the model?
    '''),
        mo.callout(
            mo.md(r'''
    **This is a live research question, not a settled one.** *LLM-BabyBench*
    (Choukrani et al., 2025,
    [alphaXiv](https://www.alphaxiv.org/abs/2505.12135)) asks it
    systematically: give models the whole map as text, fully observed, and
    they still struggle to produce a working plan, with the failure
    tracking grid size more than clutter. That sits inside a louder debate.
    Subbarao Kambhampati and colleagues argue LLMs are "approximate
    retrievers," not planners, fluent at *describing* a plan but unreliable at
    *finding* one ([Can LLMs Reason and
    Plan?](https://www.alphaxiv.org/abs/2403.04121);
    [PlanBench](https://www.alphaxiv.org/abs/2206.10498)). What you do below
    puts that claim to a live test: same worlds, a real model, your own eyes.
    '''),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def fidelity_grammar():
    mo.accordion({
        "📘 The formal language we traded away, and why ours is a different *cut*": mo.md(r"""
    The paper's missions are written in **Baby Language**, a formal, compositional grammar
    (Chevalier-Boisvert et al., 2019, Fig. 2). Verbatim:

    ```bnf
    Sent   ::= Sent1 | Sent1 ', then' Sent1 | Sent1 'after you' Sent1
    Sent1  ::= Clause | Clause 'and' Clause
    Clause ::= 'go to' Descr | 'pick up' DescrNotDoor
             | 'open' DescrDoor | 'put' DescrNotDoor 'next to' Descr
    Descr  ::= Article Color (door | ball | box | key) LocSpec
    Color  ::= ε | red | green | blue | purple | yellow | grey
    LocSpec::= ε | 'on your left' | 'on your right' | 'in front of you' | 'behind you'
    ```

    Four verbs (`go to`, `pick up`, `open`, `put … next to …`), four object types, six colors, and
    the paper counts **2.48×10¹⁹ possible instructions**. Rich, unambiguous, *generated per episode*.

    **Our cut is the opposite.** We hold the mission to **one fixed objective** and make the
    **world** the variable instead:

    | | Baby Language | Wanderland (ours) |
    |---|---|---|
    | what varies | the **instruction** (10¹⁹ of them) | the **world** (size, obstacles, gems) |
    | the objective | reach / pick / open a *named object* | collect **every** gem, reach the **exit**, avoid **lava** |
    | what it isolates | language grounding | **planning as the grid scales** |

    Neither is "more correct." For a *scaling* question (does planning collapse as the grid
    grows?), fixing the sentence and growing the world is the cleaner design: you never confound
    "harder world" with "harder sentence." And `gem`, `lava`, and a goal *tile* simply aren't
    sayable in Baby Language, so a gem/lava/goal objective was always going to be bespoke. The
    `wanderland` library *does* implement the full BabyAI vocabulary (`ball`/`box`/`pickup`/
    `drop`/`toggle`); we just point it at a different game.
    """)
    })
    return


@app.cell(hide_code=True)
def fidelity_prompt():
    paper_note(
        "The description we hand the model is our own format: a structured coordinate "
        "listing, plus an optional ASCII grid view we invented. The paper uses a different "
        "template (rooms, an object list, a Baby-Language mission); neither is the other's. And we "
        "pose only the paper's <b>Plan</b> task here, not Predict (say what a single action "
        "does) or Decompose (break a goal into ordered subgoals)."
    )
    return


@app.cell(hide_code=True)
def gpu_stats():
    _fields = ["name", "memory.total", "memory.used", "memory.free", "utilization.gpu", "temperature.gpu"]
    _r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(_fields)}", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if _r.returncode != 0:
        gpu_panel = mo.callout(mo.md("**No GPU detected.** The model cells need a CUDA box."), kind="warn")
    else:
        _hdr = ["Name", "Mem total", "Mem used", "Mem free", "GPU util", "Temp"]
        _rows = [dict(zip(_hdr, line.split(", "))) for line in _r.stdout.strip().splitlines()]
        gpu_panel = mo.ui.table(_rows, selection=None, label="GPU (nvidia-smi)")
    gpu_panel
    return


@app.cell(hide_code=True)
def llm_config():
    # Model menu: pulled straight from featherlm.MODELS, so models you add to the library
    # show up here automatically. Nothing loads by default; a pick loads it in-kernel.
    MODEL_MENU = {_k: _v for _k, _v in featherlm.MODELS.items() if _v.get("kind") == "bf16"}
    model_select = mo.ui.dropdown(
        options=["Pick a model to load…"] + list(MODEL_MENU.keys()),
        value="Pick a model to load…", label="Model")
    LLM_DEVICE = "cuda"
    llm_seed = mo.ui.number(start=0, stop=99999, value=3407, label="🎲 LLM seed")
    mo.vstack([
        mo.md("Pick a model and it loads **in-kernel** on the RTX PRO 6000 **only when you "
              "select it** (nothing loads by default). The menu is **featherlm's own model "
              "list**, so it grows as the library does, from a 4B up to a **117B** gpt-oss "
              "(and a 235B if you're patient). Switching reloads."),
        model_select,
        mo.md("🎲 **Seed** &middot; pins the model's sampling so a run is identical "
              "reload-after-reload. Change it to explore the *distribution* of how the "
              "model behaves."),
        llm_seed,
    ])
    return MODEL_MENU, llm_seed, model_select


@app.cell(hide_code=True)
def llm_load(MODEL_MENU, model_select):
    _sel = model_select.value

    import time as _time, warnings as _warnings

    mo.stop(not _sel or _sel == "Pick a model to load…",
            mo.md("⬆️ **Pick a model** in the dropdown above to load it on the GPU."))

    _spec = MODEL_MENU[_sel]
    MODEL_ID = _spec["id"]
    _kind = _spec.get("kind")

    # Free EVERY previously loaded model before loading a new one (meta-device trick).
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for _obj in list(gc.get_objects()):
            try:
                if isinstance(_obj, PreTrainedModel):
                    _obj.to("meta")
            except Exception:
                pass
    globals().pop("llm", None)
    globals().pop("tokenizer", None)
    gc.collect()
    torch.cuda.empty_cache()

    # The 235B GPTQ-offload path needs gptqmodel; install it lazily the first time it's picked.
    if _kind == "gptq_offload":
        import sys as _sys_g, importlib.util as _ilu_g
        if _ilu_g.find_spec("gptqmodel") is None:
            with mo.status.spinner(title="Installing gptqmodel for the offload path…"):
                subprocess.run([_sys_g.executable, "-m", "pip", "install", "-q", "gptqmodel"])

    # featherlm auto-picks the fastest path; bf16 warms the compiler (~1 min) then is fast.
    _t = _time.time()
    with mo.status.spinner(title=f"Loading {MODEL_ID} via featherlm…"):
        llm = featherlm.load(MODEL_ID, kind=_kind)
        tokenizer = llm.tok
    mo.md(
        f"Loaded `{MODEL_ID}` &middot; family `{llm.fam}` &middot; config `{llm.kind}` in "
        f"**{_time.time() - _t:.1f}s** &middot; VRAM "
        f"**{torch.cuda.memory_allocated() // 1024**3} GiB** / 96"
    )
    return MODEL_ID, llm


@app.cell(hide_code=True)
def llm_connector():
    def build_planner_prompt(env):
        """The exact text the model sees: the structured world + its action space.
        ONE fixed, complete prompt for every world (lava + unambiguous action
        mechanics + non-presuppositional gems). Model-agnostic."""
        lines = "\n".join(f"- {a['name']}: {a['doc']}" for a in env.actions_doc)
        valid = ", ".join(a["name"] for a in env.actions_doc)
        base = env.to_prompt("structured")
        _lava = sorted(env._puzzle.lava_set, key=lambda cr: (cr[1], cr[0]))
        if _lava:
            base += (
                "\nLava (you CAN move onto these tiles, but stepping on lava "
                "KILLS you instantly -- never step here): "
                + ", ".join(f"({c},{r})" for c, r in _lava)
            )
        return (
            "You are an agent in a 2D grid world. Plan a sequence of actions to "
            "collect every gem the world contains and finish on the goal tile, "
            "without ever stepping on lava. A solution counts ONLY when every gem has been collected AND you finish standing on the goal tile -- reaching the goal tile with any gem still uncollected does NOT count (if the world has no gems, you only need to reach the goal). Coordinates are (x, y) with x going "
            "east and y going south from the top-left.\n\n"
            "How your actions work (the map's y points DOWN, so don't assume the "
            "usual math orientation): your facing is one of north/east/south/west. "
            "move_forward changes position by north=(x, y-1), east=(x+1, y), "
            "south=(x, y+1), west=(x-1, y). turn_right cycles your facing "
            "north->east->south->west->north; turn_left cycles the reverse "
            "(north->west->south->east->north).\n\n"
            f"{base}\n\n"
            f"Allowed actions:\n{lines}\n\n"
            "Output ONLY your final plan as a JSON array of action strings from: "
            f'{valid}\nExample: ["turn_right", "move_forward", "move_forward"]'
        )

    def parse_plan(text, valid):
        """Pull the COMMITTED plan from any model's reply: the LAST JSON array of
        valid action names. Returns [] if none is present yet (e.g. the model is
        still thinking). We deliberately do NOT token-scan loose prose -- a
        reasoning trace is full of action words and would parse into a bogus,
        flickering 'plan' mid-stream (the very bug this avoids)."""
        valid = set(valid)
        for arr in reversed(re.findall(r"\[[^\[\]]*\]", text, re.S)):
            try:
                picked = [str(x).strip() for x in json.loads(arr) if str(x).strip() in valid]
                if picked:
                    return picked
            except Exception:
                pass
        return []


    return build_planner_prompt, parse_plan


@app.cell(hide_code=True)
def act3_ask_button():
    reason_toggle = mo.ui.switch(value=False, label="🧠 thinking")
    think_budget = mo.ui.slider(1024, 16384, value=8192, step=1024, label="thinking budget (tokens)")
    ask_btn = mo.ui.run_button(label="Plan this world")
    return ask_btn, reason_toggle, think_budget


@app.cell(hide_code=True)
def act3_controls_view(ask_btn, reason_toggle, think_budget):
    mo.hstack(
        [ask_btn, reason_toggle] + ([think_budget] if reason_toggle.value else []),
        justify="start", gap=1.2,
    )
    return


@app.cell(hide_code=True)
def act3_attempt(
    MODEL_ID,
    ask_btn,
    build_planner_prompt,
    gen_env,
    gen_puzzle,
    llm,
    llm_seed,
    parse_plan,
    reason_toggle,
    think_budget,
):
    mo.stop(not ask_btn.value, mo.md("_Tweak a world in Act 2, then press the button to send it to the model._"))
    mo.stop(not MODEL_ID, mo.md("⬆️ Load a model first (pick one in the dropdown above)."))

    _env = gen_env
    _valid = [a["name"] for a in _env.actions_doc]
    _oracle = bp.solve(gen_puzzle)
    _think = reason_toggle.value
    _budget = think_budget.value


    def _scroll(text, h=320):
        _safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        _js = ("navigator.clipboard.writeText(this.nextElementSibling"
               ".querySelector('.rtxt').innerText);this.textContent='✓ Copied'")
        return mo.Html(
            f'<div style="position:relative">'
            f'<button onclick="{_js}" style="position:absolute;top:6px;right:10px;'
            f'z-index:3;font-size:0.7rem;padding:2px 9px;border:1px solid '
            f'rgba(0,0,0,0.15);border-radius:6px;background:var('
            f'--marimo-background-color,#fff);cursor:pointer">📋 Copy</button>'
            f'<div style="'
            f'border:1px solid rgba(0,0,0,0.12);border-radius:10px;'
            f'background:rgba(0,0,0,0.03)">'
            f'<div class="rtxt" style="white-space:pre-wrap;font-family:ui-monospace,'
            f'SFMono-Regular,Menlo,monospace;font-size:0.8rem;line-height:1.5;'
            f'padding:0.7rem 0.9rem">{_safe}</div></div></div>'
        )


    def _chips(actions, busy):
        if not actions:
            return mo.md("🧠 _thinking… (no plan committed yet)_" if busy
                         else "_No plan found in the reply._")
        return mo.vstack([
            mo.md(f"Plan · {len(actions)} actions"),
            mo.md("  ".join(f"`{a}`" for a in actions[-70:])),
        ])


    def _live(reasoning, actions, busy):
        _b = []
        if reasoning:
            _b.append(mo.md("🧠 Reasoning " + ("*(streaming…)*" if busy else "")))
            _b.append(_scroll(reasoning, 320))
        _b.append(_chips(actions, busy))
        return mo.vstack(_b)


    # featherlm streams the generation for us; events carry the live (reasoning, answer)
    # split, so the panel and the committed-plan chips update without any manual decode.
    mo.output.replace(_live("", [], True))
    _ev = {"reasoning": "", "answer": ""}
    _tick = 0
    for _ev in llm.stream(build_planner_prompt(_env), thinking=_think,
                          max_tokens=(_budget if _think else None), seed=int(llm_seed.value)):
        _tick += 1
        if _tick % 16 == 0:
            mo.output.replace(_live(_ev["reasoning"], parse_plan(_ev["answer"], _valid), True))

    _plan = parse_plan(_ev["answer"], _valid)
    _r = _ev["reasoning"]
    _attempt_env = bp.World(gen_puzzle)
    _res = _attempt_env.act(_plan)
    _scene = mo.ui.anywidget(_attempt_env)
    _status = "🎉 solved it!" if _res["success"] else (
        "💀 walked into lava" if _res.get("died") else "didn't solve it")
    _out = []
    if _r:
        _out += [mo.md("🧠 Reasoning"), _scroll(_r, 420)]
    _out += [
        _chips(_plan, False),
        _scene,
        mo.callout(
            mo.md(
                f"""
                The model's attempt: {_status} &nbsp; {"(with reasoning)" if _think else "(no reasoning)"}

                - plan length: {len(_plan)} actions &nbsp;(oracle solved in {len(_oracle)})
                - gems: {_res['gems_collected']} / {_res['total_gems']} &middot;
                  reached goal: {_res['reached_goal']}

                Press Run My Code in the scene to watch it play out.
                """
            ),
            kind="success" if _res["success"] else "warn",
        ),
    ]
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def act4_intro():
    mo.vstack([
        mo.md(r'''

    ## Act 4 · When planning doesn't scale, write a planner

    You just climbed a ladder: small models flail, and only a large one, thinking hard, plans even *this* little world by hand. But reasoning it out step by step **doesn't scale**: make the map bigger and it falls apart, more tiles to track and more turns to lose the thread. Our plain BFS oracle barely notices: it just searches a slightly bigger graph, in milliseconds.

    So here's a different idea. What if the model doesn't *plan the world* at all, but instead writes a program that does? Give it a sandbox, let it write its own little oracle, run that code, and read back the plan. Its job shrinks from *"trace every step perfectly in my head"* to *"write a search, once"*, a cost that barely grows with the map.

    That move, from the model *being* the solver to the model *writing* the solver and calling it, is exactly why the industry pivoted to tool-calling and agents.
    '''),
        mo.callout(mo.md(r'''
    The field keeps arriving at the same move from different directions: when a job won't fit in the model's head, offload it instead of scaling up the thinking. [PAL](https://www.alphaxiv.org/abs/2211.10435) offloads arithmetic to a Python interpreter, [CodeAct](https://www.alphaxiv.org/abs/2402.01030) makes executable code the agent's whole action space, and [Recursive Language Models](https://www.alphaxiv.org/abs/2512.24601) put the context itself in a REPL the model re-queries. Offloading the judgment works the same way: a model [can't reliably grade its own plan](https://www.alphaxiv.org/abs/2402.08115), so you pair it with a sound external checker, the idea behind Kambhampati's [LLM-Modulo](https://www.alphaxiv.org/abs/2402.01817) framework, and the engine under [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) and [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5), where the model proposes a program and a sound judge checks it. Here we offload the planning: the model writes the search it kept losing by hand, and Mo's simulator, not the model's word, decides whether the plan reaches the goal.

    Scale helps, but it doesn't substitute: even o1 drops from ~98% to ~24% on long block-stacking plans (Valmeekam et al., TMLR 2025). That's why the models after it, o3 and o4-mini, were trained to run Python inside their own reasoning rather than trace every step out in tokens.
    '''), kind="neutral"),
    ])
    return


@app.function(hide_code=True)
def world_to_dict(env):
    """Flatten a Wanderland world into a CODE-FRIENDLY dict the model uses
    directly: coords are (x, y) tuples; walls/water/lava are sets of tuples
    (so `(x,y) in world['walls']` and `(x,y) == world['goal']` work)."""
    p = env._puzzle
    _h = {"N": 0, "E": 1, "S": 2, "W": 3}[p.heading] if isinstance(p.heading, str) else p.heading
    _objs = [
        {"pos": (c, r), "type": getattr(o, "type", None),
         "color": getattr(o, "color", None), "blocking": bool(getattr(o, "blocking", False))}
        for (c, r), o in p.objects.items()
    ]
    return {
        "cols": p.cols, "rows": p.rows,
        "start": tuple(p.start), "heading": _h,
        "goal": tuple(p.goal) if p.goal else None,
        "walls": {tuple(x) for x in p.wall_set},
        "water": {tuple(x) for x in p.gap_set},
        "lava": {tuple(x) for x in p.lava_set},
        "objects": _objs,
    }


@app.cell(hide_code=True)
def act4_agent_core(build_planner_prompt):
    import re as _re4


    def build_agent_prompt(env):
        """Act 3's world + mechanics + goal, plus: use the tool, the predefined `world`
        variable (exact types), AND the full object/door/key schema so the code-writer
        never has to guess the data format."""
        return build_planner_prompt(env) + (
            "\n\n---\n\n"
            "Do NOT work the path out by hand. Use the **run_python** tool to find the "
            "plan: write code that solves the world (search, brute force -- your choice) "
            "and prints the action list as a JSON array.\n\n"
            "The world is ALREADY loaded as a Python variable `world` in the sandbox -- "
            "USE IT DIRECTLY, do NOT retype the data. Its exact types: `world['start']` "
            "and `world['goal']` are (x, y) tuples; `world['walls']`, `world['water']`, "
            "`world['lava']` are sets of (x, y) tuples (so `(x, y) in world['walls']` and "
            "`(x, y) == world['goal']` compare correctly); `world['heading']` is "
            "0=N 1=E 2=S 3=W; `world['cols']`, `world['rows']` are ints. "
            "`world['objects']` is a list of dicts, each with `'pos'` (an (x, y) tuple), "
            "`'type'` (one of `'gem'`, `'key'`, `'door'`), `'color'` (a string like "
            "`'red'`, or `None` for gems), and `'blocking'` (a bool -- True means the tile "
            "is IMPASSABLE until the object is opened/removed; keys and LOCKED doors are "
            "blocking, gems are not).\n\n"
            "The world's rules your code must model: `move_forward` is blocked by walls, "
            "water, lava, AND any object with `blocking=True` on the target tile. "
            "**pickup** takes the key on the tile you FACE into your one hand (carry "
            "limit 1). **toggle** on a LOCKED door you FACE, while holding a key of the "
            "SAME `color`, unlocks it (that tile becomes passable). **collect_gem** "
            "collects the gem on the tile you STAND on. A solution collects EVERY gem and "
            "finishes on the goal tile.\n\n"
            "Once the tool produces a working plan, reply with ONLY the final JSON array of actions."
        )

    def parse_plan_loose(text, valid):
        """Pull a plan from sandbox stdout / a final answer, tolerant of UNQUOTED
        lists like [move_forward, turn_left, ...] that models often print. Safe on
        stdout and the post-<think> answer (no reasoning prose to mis-parse)."""
        _vs = set(valid)
        for _inner in reversed(_re4.findall(r"\[(.*?)\]", text, _re4.S)):
            _toks = [t for t in _re4.findall(r"[a-z_]+", _inner) if t in _vs]
            if _toks:
                return _toks
        return []


    return build_agent_prompt, parse_plan_loose


@app.cell(hide_code=True)
def act4_controls():
    act4_btn = mo.ui.run_button(label="Plan this world")
    act4_code = mo.ui.switch(value=True, label="⌨️ coding sandbox tool", disabled=True)
    act4_think = mo.ui.switch(value=False, label="🧠 thinking")
    act4_verify = mo.ui.switch(value=True, label="🔁 Close the loop")
    mo.hstack([act4_btn, act4_code, act4_think, act4_verify], justify="start", gap=1.2)
    return act4_btn, act4_think, act4_verify


@app.cell(hide_code=True)
def act4_attempt(
    MODEL_ID,
    act4_btn,
    act4_think,
    act4_verify,
    build_agent_prompt,
    gen_env,
    gen_puzzle,
    llm,
    llm_seed,
    parse_plan,
    parse_plan_loose,
):
    mo.stop(not act4_btn.value, mo.md("_Press Plan this world to run._"))
    mo.stop(not MODEL_ID, mo.md("⬆️ Load a model in Act 3's dropdown first."))

    import re as _re_a4

    _env = gen_env
    _world = world_to_dict(_env)
    _valid = [a["name"] for a in _env.actions_doc]
    _oracle = bp.solve(gen_puzzle)
    _think = act4_think.value
    _verify = act4_verify.value
    _fmt = llm.fam
    _max_turns = 6


    def _box(text, h=240):
        _s = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return mo.Html(
            f'<div style="white-space:pre-wrap;'
            "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.8rem;"
            "line-height:1.5;padding:0.6rem 0.85rem;border:1px solid rgba(0,0,0,0.12);"
            f'border-radius:8px;background:rgba(0,0,0,0.03)">{_s}</div>'
        )


    def _clean(text):
        if _fmt != "gemma":
            return text
        text = _re_a4.sub(r"<\|channel>\w*<channel\|>", "", text)
        text = text.replace('<|"|>', '"')
        for _t in ("<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|turn>", "<turn|>"):
            text = text.replace(_t, "")
        return text


    _log = []


    def _view(live=None, label=""):
        parts = [mo.md("### 🤖 The model writes code")] + list(_log)
        if live is not None:
            parts.append(mo.md(f"{label} *(writing…)*"))
            parts.append(_box(live, 300))
        return mo.vstack(parts)


    mo.output.replace(_view())
    _messages = [{"role": "user", "content": build_agent_prompt(_env)}]
    _final_plan = []
    for _turn in range(_max_turns):
        # featherlm.chat streams a tool-aware turn and returns the (reasoning, answer)
        # split; tool-call markers live in `answer`, which we append as the assistant turn.
        _ev = {"reasoning": "", "answer": "", "raw": ""}
        _tick = 0
        for _ev in llm.chat(_messages, tools=[featherlm.python_tool()], thinking=_think,
                            max_tokens=(6144 if _think else 4096),
                            seed=int(llm_seed.value) + _turn, stream=True):
            _tick += 1
            if _tick % 16 == 0:
                mo.output.replace(_view(_clean(_ev["raw"]), f"Turn {_turn + 1}"))
        _reasoning, _answer = _ev["reasoning"], _ev["answer"]
        _messages.append({"role": "assistant", "content": _answer})
        if _reasoning:
            _log.append(mo.md(f"Turn {_turn + 1} · 🧠 reasoning"))
            _log.append(_box(_reasoning, 220))
        _calls = featherlm.parse_tool_calls(_answer, llm.fam)
        _via = "native tool call"
        if not _calls:
            _c = featherlm.extract_code(_answer)
            if _c:
                _calls = [{"name": "run_python", "arguments": {"code": _c}}]
                _via = "code block"
        if _calls:
            _solved = False
            for _call in _calls:
                _code = (_call.get("arguments") or {}).get("code", "")
                _res = featherlm.run_python(_code, env={"world": _world})
                _out = (_res["stdout"] + ("\n" + _res["stderr"] if _res["stderr"] else "")).strip()
                _ptry = parse_plan_loose(_res["stdout"], _valid) if _res["ok"] else []
                _sc = bp.World(gen_puzzle).act(_ptry) if _ptry else None
                _log.append(mo.md(f"Turn {_turn + 1} · 🛠️ wrote code ({_via})"))
                _log.append(mo.md("```python\n" + _code[:6000] + "\n```"))
                _log.append(mo.md("sandbox output →"))
                _log.append(_box(_out or "(no output)", 150))
                _fb = "Sandbox output:\n```\n" + (_out[:3000] or "(no output)") + "\n```\n"
                if _ptry and _sc:
                    if _sc["success"]:
                        _log.append(mo.callout(mo.md(f"✅ verifier: that plan solves it · {_sc['gems_collected']}/{_sc['total_gems']} gems, goal reached.{'' if _verify else ' *(not shared with the model)*'}"), kind="success"))
                    else:
                        _log.append(mo.callout(mo.md(f"❌ verifier: not solved · gems {_sc['gems_collected']}/{_sc['total_gems']}, goal {_sc['reached_goal']}, died {_sc['died']}.{' Sent back to the model.' if _verify else ' *(not shared with the model)*'}"), kind="warn"))
                    if _verify:
                        if _sc["success"]:
                            _fb += "I ran that plan in the world and it SOLVES it. Reply with ONLY the final JSON array of actions."
                        else:
                            _fb += f"I ran that plan in the world: reached_goal={_sc['reached_goal']}, gems={_sc['gems_collected']}/{_sc['total_gems']}, died={_sc['died']}. It does NOT solve the world. Remember: EVERY gem must be collected AND you must finish on the goal tile -- a path that reaches the goal without collecting all gems does not count. Fix your code and run it again."
                    else:
                        _fb += "If that is your final plan, reply with ONLY the JSON array of actions; otherwise revise your code and run it again."
                else:
                    _log.append(mo.callout(mo.md("⚠️ no plan: the code errored or found no solution (no action list printed)."), kind="warn"))
                    _fb += ("Your code did NOT print a valid plan (a JSON list of action names). It "
                            "either raised an error or found no solution. Read the output above, debug "
                            "your code, and run it again.")
                _messages.append({"role": "tool", "name": "run_python", "content": _fb})
                if _verify and _sc and _sc["success"]:
                    _final_plan = _ptry
                    _log.append(mo.md(f"Turn {_turn + 1} · ✅ verified solution accepted · {len(_ptry)} actions"))
                    _solved = True
                    break
            mo.output.replace(_view())
            if _solved:
                break
        else:
            _cand = parse_plan_loose(_answer, _valid)
            _csc = bp.World(gen_puzzle).act(_cand or [])
            if _verify and not _csc["success"] and _turn < _max_turns - 1:
                _log.append(mo.md(f"Turn {_turn + 1} · proposed a final plan · {len(_cand)} actions"))
                _log.append(mo.callout(mo.md(
                    f"❌ verifier: that plan does not solve it · gems "
                    f"{_csc['gems_collected']}/{_csc['total_gems']}, goal {_csc['reached_goal']}, "
                    f"died {_csc['died']}. Sent back, keep fixing."), kind="warn"))
                _messages.append({"role": "user", "content":
                    f"I ran that plan in the world: reached_goal={_csc['reached_goal']}, "
                    f"gems={_csc['gems_collected']}/{_csc['total_gems']}, died={_csc['died']}. It does "
                    "NOT solve the world. Use the run_python tool to fix your code and find a plan "
                    "that collects ALL gems AND reaches the goal."})
                continue
            _final_plan = _cand
            _log.append(mo.md(f"Turn {_turn + 1} · ✅ final answer · {len(_final_plan)} actions"))
            break

    if not _final_plan:
        _final_plan = parse_plan(_messages[-1].get("content", ""), _valid)

    _attempt_env = bp.World(gen_puzzle)
    _resf = _attempt_env.act(_final_plan or [])
    _scene = mo.ui.anywidget(_attempt_env)
    mo.output.replace(mo.vstack(list(_log) + [
        mo.md("### Result · Mo plays the model's plan"),
        _scene,
        mo.callout(
            mo.md(
                f"The model's code {'SOLVES' if _resf['success'] else 'does not solve'} the world. "
                f"plan: {len(_final_plan)} actions (oracle {len(_oracle)}) &middot; "
                f"gems {_resf['gems_collected']}/{_resf['total_gems']} &middot; "
                f"reached goal {_resf['reached_goal']}"
            ),
            kind="success" if _resf["success"] else "warn",
        ),
    ]))
    return


@app.cell(hide_code=True)
def act5_intro():
    mo.md(r"""
    ## Act 5 · Does it actually scale?
    """)
    return


@app.cell(hide_code=True)
def act5_banner():
    mo.Html(r"""<style>@keyframes oracle_a5pulse{0%,100%{opacity:.7}50%{opacity:1}}@keyframes oracle_a5shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-2px)}75%{transform:translateX(2px)}}@keyframes oracle_a5spark{0%,100%{opacity:.3;transform:scale(.7)}50%{opacity:1;transform:scale(1.1)}}@keyframes oracle_a5no{0%,100%{transform:rotate(-12deg)}15%{transform:rotate(-20deg)}38%{transform:rotate(-4deg)}62%{transform:rotate(-20deg)}85%{transform:rotate(-4deg)}}@keyframes oracle_a5yes{0%,100%{transform:translateY(0) rotate(4deg)}25%{transform:translateY(3px) rotate(2deg)}50%{transform:translateY(-1px) rotate(6deg)}75%{transform:translateY(3px) rotate(2deg)}}@keyframes oracle_a5fill0{0%,4.0%{opacity:0}6.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill1{0%,6.0%{opacity:0}8.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill2{0%,8.0%{opacity:0}10.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill3{0%,10.0%{opacity:0}12.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill4{0%,12.0%{opacity:0}14.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill5{0%,14.0%{opacity:0}16.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill6{0%,16.0%{opacity:0}18.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill7{0%,18.0%{opacity:0}20.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill8{0%,20.0%{opacity:0}22.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill9{0%,22.0%{opacity:0}24.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill10{0%,24.0%{opacity:0}26.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill11{0%,26.0%{opacity:0}28.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill12{0%,28.0%{opacity:0}30.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill13{0%,30.0%{opacity:0}32.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill14{0%,32.0%{opacity:0}34.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill15{0%,34.0%{opacity:0}36.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill16{0%,36.0%{opacity:0}38.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill17{0%,38.0%{opacity:0}40.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill18{0%,40.0%{opacity:0}42.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill19{0%,42.0%{opacity:0}44.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill20{0%,44.0%{opacity:0}46.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill21{0%,46.0%{opacity:0}48.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill22{0%,48.0%{opacity:0}50.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill23{0%,50.0%{opacity:0}52.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill24{0%,52.0%{opacity:0}54.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill25{0%,54.0%{opacity:0}56.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill26{0%,56.0%{opacity:0}58.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill27{0%,58.0%{opacity:0}60.0%,90%{opacity:1}100%{opacity:0}}@keyframes oracle_a5fill28{0%,60.0%{opacity:0}62.0%,90%{opacity:1}100%{opacity:0}}</style><div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;background:#15171c;width:100%;max-width:540px;margin:0 auto;border-radius:12px;overflow:hidden;box-sizing:border-box"><div style="background:#f4b81f;display:flex;align-items:center;justify-content:center;gap:10px;padding:14px 12px;min-height:170px;box-sizing:border-box"><div style="display:inline-block;transform-origin:bottom center;transform:rotate(-10deg)"><pre style="font:11px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">             <span style="color:rgb(124,140,66)">█</span><span style="color:rgb(155,172,85)">█</span>  
                <span style="color:rgb(124,140,64)">█</span><span style="color:rgb(95,110,51)">█</span> <span style="color:rgb(155,172,85)">█</span><span style="color:rgb(155,172,86)">█</span>
             <span style="color:rgb(155,172,87)">█</span>   <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(157,171,85)">█</span><span style="color:rgb(155,172,85)">█</span> 
        <span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span>    
      <span style="color:rgb(94,111,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,66)">█</span>   
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(155,172,85)">███</span><span style="color:rgb(124,140,64)">█</span>  
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">█████████</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,64)">██</span> 
    <span style="color:rgb(95,111,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span> 
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(95,110,50)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(124,140,65)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(128,40,5)">█</span><span style="color:rgb(163,51,6)">██████████████</span><span style="color:rgb(126,39,5)">█</span> 
    <span style="color:rgb(128,40,5)">█</span><span style="color:rgb(126,39,5)">█</span><span style="color:rgb(163,51,6)">████████████</span><span style="color:rgb(126,39,5)">██</span> 
     <span style="color:rgb(126,39,5)">█</span><span style="color:rgb(163,51,6)">████████████</span><span style="color:rgb(126,39,5)">█</span>  
      <span style="color:rgb(126,39,5)">█</span><span style="color:rgb(163,51,6)">█</span>  <span style="color:rgb(163,51,6)">███</span>  <span style="color:rgb(163,51,6)">██</span>    </pre></div></div><div style="background:#ececef;display:flex;align-items:center;justify-content:start;gap:10px;padding:14px 12px;min-height:170px;box-sizing:border-box"><span style="display:inline-block;animation:oracle_a5pulse 2s ease-in-out infinite"><pre style="font:4px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">              <span style="color:rgb(0,0,0)">█████</span>               
            <span style="color:rgb(0,0,0)">██████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██████</span>         
         <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(254,150,223)">█████████████████</span><span style="color:rgb(0,0,0)">███</span>      
       <span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██</span>    
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(197,134,253)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">███████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███████████████████</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██████████</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█████</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(162,168,249)">█</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(162,168,249)">██</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(162,168,249)">██</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span>
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">██</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>  
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">██████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">████████</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█████████████████████</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█████████████████</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>    
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█████████████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">█</span>   </pre></span><div style="display:flex;align-items:center;gap:10px"><span style="color:#bbb;font-size:9px">••</span><div style="background:#fff;border:2px solid #cdd1d8;border-radius:14px;width:96px;height:82px;display:flex;align-items:center;justify-content:center"><div style="width:90px;height:76px;display:flex;align-items:center;justify-content:center"><div style="transform:rotateX(56deg) rotateZ(-45deg);transform-origin:center center"><pre style="font:9px/0.78 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0"><span style="color:rgb(228,66,58);animation:oracle_a5fill0 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span>
    <span style="color:rgb(228,66,58);animation:oracle_a5fill1 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span>
    <span style="color:rgb(228,66,58);animation:oracle_a5fill2 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill3 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill4 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill10 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill11 4.0s linear infinite">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span>
    <span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill5 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill9 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill12 4.0s linear infinite">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span>
    <span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill6 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill7 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill8 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill13 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span>
    <span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill14 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill15 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill16 4.0s linear infinite">█</span><span style="color:rgb(228,66,58);animation:oracle_a5fill17 4.0s linear infinite">█</span><span style="color:rgb(58,62,72)">█</span>
    <span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(140,144,156)">▣</span>
    <span style="color:rgb(44,48,57)">█</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(140,144,156)">▣</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span><span style="color:rgb(44,48,57)">█</span><span style="color:rgb(58,62,72)">█</span></pre></div></div></div></div></div><div style="background:#f4b81f;display:flex;align-items:center;justify-content:center;gap:10px;padding:14px 12px;min-height:170px;box-sizing:border-box"><div style="display:inline-block;transform-origin:bottom center;animation:oracle_a5yes 1.8s ease-in-out infinite"><div style="display:flex;align-items:flex-start;gap:4px"><span style="animation:oracle_a5spark 1.8s ease-in-out infinite"><pre style="font:7px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;"> <span style="color:rgb(255,228,90)">█</span> 
    <span style="color:rgb(255,228,90)">███</span>
     <span style="color:rgb(255,228,90)">█</span> </pre></span><pre style="font:11px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">             <span style="color:rgb(124,140,66)">█</span><span style="color:rgb(155,172,85)">█</span>  
                <span style="color:rgb(124,140,64)">█</span><span style="color:rgb(95,110,51)">█</span> <span style="color:rgb(155,172,85)">█</span><span style="color:rgb(155,172,86)">█</span>
             <span style="color:rgb(155,172,87)">█</span>   <span style="color:rgb(94,111,48)">█</span><span style="color:rgb(157,171,85)">█</span><span style="color:rgb(155,172,85)">█</span> 
        <span style="color:rgb(124,140,64)">██████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span>    
      <span style="color:rgb(94,111,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(155,172,84)">█</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,66)">█</span>   
     <span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">████████</span><span style="color:rgb(155,172,85)">███</span><span style="color:rgb(124,140,64)">█</span>  
    <span style="color:rgb(95,110,51)">█</span><span style="color:rgb(95,110,48)">█</span><span style="color:rgb(124,140,63)">█</span><span style="color:rgb(124,140,64)">█████████</span><span style="color:rgb(155,172,85)">██</span><span style="color:rgb(124,140,64)">██</span> 
    <span style="color:rgb(95,111,49)">█</span><span style="color:rgb(124,140,64)">████</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(124,140,64)">███</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(124,140,64)">██</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span> 
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████████████</span><span style="color:rgb(95,110,50)">█</span>
    <span style="color:rgb(95,110,49)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(124,140,65)">█</span><span style="color:rgb(124,140,64)">███████</span><span style="color:rgb(94,111,48)">█</span>
    <span style="color:rgb(128,40,5)">█</span><span style="color:rgb(163,51,6)">██████████████</span><span style="color:rgb(126,39,5)">█</span> 
    <span style="color:rgb(128,40,5)">█</span><span style="color:rgb(126,39,5)">█</span><span style="color:rgb(163,51,6)">████████████</span><span style="color:rgb(126,39,5)">██</span> 
     <span style="color:rgb(126,39,5)">█</span><span style="color:rgb(163,51,6)">████████████</span><span style="color:rgb(126,39,5)">█</span>  
      <span style="color:rgb(126,39,5)">█</span><span style="color:rgb(163,51,6)">█</span>  <span style="color:rgb(163,51,6)">███</span>  <span style="color:rgb(163,51,6)">██</span>    </pre></div></div></div><div style="background:#ececef;display:flex;align-items:center;justify-content:start;gap:10px;padding:14px 12px;min-height:170px;box-sizing:border-box"><span style="display:inline-block;animation:oracle_a5pulse 2s ease-in-out infinite"><pre style="font:4px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">              <span style="color:rgb(0,0,0)">█████</span>               
            <span style="color:rgb(0,0,0)">██████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██████</span>         
         <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(254,150,223)">█████████████████</span><span style="color:rgb(0,0,0)">███</span>      
       <span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██</span>    
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(197,134,253)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(0,0,0)">█████</span><span style="color:rgb(254,150,223)">███████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███████████████████</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██████████</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█████</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">████</span><span style="color:rgb(162,168,249)">█</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(162,168,249)">██</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">██</span><span style="color:rgb(162,168,249)">██</span><span style="color:rgb(89,101,244)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">█████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(89,101,244)">███</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>
    <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█████████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span>
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██████</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">██</span> 
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>  
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(0,0,0)">█</span> <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">██</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>  
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">████</span><span style="color:rgb(0,0,0)">██████</span><span style="color:rgb(254,150,223)">███</span><span style="color:rgb(0,0,0)">████████</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(197,134,253)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>   
      <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█████████████████████</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span>   
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">███</span><span style="color:rgb(254,150,223)">█████████████████</span><span style="color:rgb(253,96,206)">██</span><span style="color:rgb(169,78,252)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">█</span>    
     <span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(254,150,223)">█</span><span style="color:rgb(253,96,206)">█</span><span style="color:rgb(0,0,0)">███</span><span style="color:rgb(253,96,206)">█████████████████</span><span style="color:rgb(0,0,0)">█</span><span style="color:rgb(253,96,206)">████</span><span style="color:rgb(0,0,0)">█</span>   </pre></span><div style="display:flex;align-items:center;gap:10px"><span style="color:#bbb;font-size:9px">••</span><div style="background:#fff;border:2px solid #cdd1d8;border-radius:14px;width:96px;height:82px;display:flex;align-items:center;justify-content:center"><pre style="font:8px/0.68 ui-monospace,monospace;background:transparent;margin:0;letter-spacing:0;">    <span style="color:rgb(120,225,245)">██</span>    
         <span style="color:rgb(120,225,245)">█</span>    
      <span style="color:rgb(95,115,130)">██████</span>  
     <span style="color:rgb(95,115,130)">█</span><span style="color:rgb(150,175,190)">██████</span><span style="color:rgb(95,115,130)">█</span> 
    <span style="color:rgb(95,115,130)">█</span><span style="color:rgb(150,175,190)">█</span> <span style="color:rgb(45,225,255)">█</span>  <span style="color:rgb(45,225,255)">█</span> <span style="color:rgb(150,175,190)">█</span><span style="color:rgb(95,115,130)">█</span>
    <span style="color:rgb(95,115,130)">█</span><span style="color:rgb(150,175,190)">█</span>      <span style="color:rgb(150,175,190)">█</span><span style="color:rgb(95,115,130)">█</span>
    <span style="color:rgb(95,115,130)">█</span><span style="color:rgb(150,175,190)">█</span> <span style="color:rgb(70,200,235)">████</span> <span style="color:rgb(150,175,190)">█</span><span style="color:rgb(95,115,130)">█</span>
    <span style="color:rgb(95,115,130)">█</span><span style="color:rgb(150,175,190)">████████</span><span style="color:rgb(95,115,130)">█</span>
     <span style="color:rgb(95,115,130)">████████</span> 
     <span style="color:rgb(95,115,130)">█</span> <span style="color:rgb(95,115,130)">████</span> <span style="color:rgb(95,115,130)">█</span> 
     <span style="color:rgb(95,115,130)">██</span> <span style="color:rgb(95,115,130)">██</span> <span style="color:rgb(95,115,130)">██</span> 
     <span style="color:rgb(95,115,130)">█</span>      <span style="color:rgb(95,115,130)">█</span> 
    <span style="color:rgb(95,115,130)">██</span>      <span style="color:rgb(95,115,130)">██</span></pre></div></div></div></div>""")
    return


@app.cell(hide_code=True)
def act5_load():
    from datasets import load_dataset
    import datasets as _datasets
    import huggingface_hub as _hfhub

    # dataset is public -> no token; quiet the unauthenticated + progress-bar noise
    for _quiet in (
        lambda: _datasets.disable_progress_bars(),
        lambda: _datasets.logging.set_verbosity_error(),
        lambda: _hfhub.utils.logging.set_verbosity_error(),
    ):
        try:
            _quiet()
        except Exception:
            pass

    act5_df = load_dataset(
        "orbrx/wanderland-llm-planning", split="train",
        download_mode="force_redownload",
    ).to_pandas()

    _n_err = int((act5_df.harness_error.astype(str).str.strip().str.len() > 0).sum())
    _model = act5_df.model_id.iloc[0]
    _lo, _hi = int(act5_df.cols.min()), int(act5_df.cols.max())
    _seeds = act5_df.world_seed.nunique()
    _meth = act5_df.method_label.nunique()
    mo.md(
        f"As the world grows, does each approach hold up? "
        f"We ran the same approaches over a grid of worlds, live on the GPU, and streamed every "
        f"run to a [public dataset](https://huggingface.co/datasets/orbrx/wanderland-llm-planning).\n\n"
        f"📊 **{len(act5_df)} runs** loaded · {_n_err} errors · one model "
        f"(`{_model}`) · {_meth} approaches × grid {_lo}×{_lo} → {_hi}×{_hi} × "
        f"{_seeds} seeds, measuring *solved · wall-time · tokens*.  Now we let the numbers settle it."
    )
    return (act5_df,)


@app.cell(hide_code=True)
def act5_chart(act5_df):
    import altair as alt

    act5_lab = {"plan_nothink": "plan · no-think", "plan_think": "plan · think",
                "code": "write code", "code_fb": "code + verifier"}
    _dom = ["plan · no-think", "plan · think", "write code", "code + verifier"]
    _rng = ["#d4564a", "#e0912f", "#2a9d4a", "#14532d"]
    _agg = (act5_df.groupby(["method_label", "cols"])
            .agg(success=("solved", "mean"), wall=("wall_time_s", "median"),
                 tokens=("output_tokens", "median"), n=("solved", "size")).reset_index())
    _agg["approach"] = _agg.method_label.map(act5_lab)

    _tip = [alt.Tooltip("approach:N", title="approach"), alt.Tooltip("cols:Q", title="grid size"),
            alt.Tooltip("success:Q", title="solved", format=".0%"),
            alt.Tooltip("wall:Q", title="wall-time (s)", format=".0f"),
            alt.Tooltip("tokens:Q", title="tokens", format=".0f")]
    _color = alt.Color("approach:N", scale=alt.Scale(domain=_dom, range=_rng), legend=None)
    _base = alt.Chart(_agg).encode(
        x=alt.X("cols:Q", title="grid size (n × n)", scale=alt.Scale(nice=False)),
        color=_color, tooltip=_tip)
    _succ = _base.mark_line(point=alt.OverlayMarkDef(size=75, filled=True), strokeWidth=3).encode(
        y=alt.Y("success:Q", title="fraction solved", scale=alt.Scale(domain=[0, 1]))
    ).properties(title="Success (higher is better)", width=300, height=255)
    _cost = _base.mark_line(point=alt.OverlayMarkDef(size=75, filled=True), strokeWidth=3).encode(
        y=alt.Y("wall:Q", title="median wall-time (s)")
    ).properties(title="Cost (lower is better)", width=300, height=255)
    act5_chart_ui = mo.ui.altair_chart(alt.hconcat(_succ, _cost),
                                       chart_selection=False, legend_selection=False)

    _legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 10px">'
        f'<span style="width:11px;height:11px;border-radius:50%;background:{_c};'
        f'display:inline-block"></span>{_l}</span>'
        for _l, _c in zip(_dom, _rng)
    )
    mo.vstack([
        mo.Html("<div style='text-align:center;line-height:1.3'>"
                "<b>Success and compute cost vs. grid size, by planning strategy</b><br>"
                "<span style='font-size:0.85em;opacity:0.7'>Gemma-4-31B-it · "
                "median over 3 seeds per grid size</span></div>"),
        mo.Html(f'<div style="display:flex;justify-content:center;flex-wrap:wrap;'
                f'font-size:13px;margin:2px 0 4px">{_legend}</div>'),
        mo.center(act5_chart_ui),
    ])
    return (act5_lab,)


@app.cell(hide_code=True)
def act5_drill(act5_df, act5_lab):
    _sizes = sorted(act5_df.cols.unique())
    _opts = {f"{lab} · {s}×{s}": f"{m}|{s}" for m, lab in act5_lab.items() for s in _sizes}
    act5_inspect_pick = mo.ui.dropdown(options=_opts, value=None,
                                       label="🔍 inspect a run:")
    mo.accordion({
        "🔍 Curious about any of the data points? Replay one of these runs in Mo's world": act5_inspect_pick
    })
    return (act5_inspect_pick,)


@app.cell(hide_code=True)
def act5_inspect(act5_df, act5_inspect_pick, act5_lab, random_world):
    if act5_inspect_pick.value is None:
        _out = mo.md("Open the panel above and pick a run, and it'll replay here in Mo's world.")
    else:
        _m, _sz = act5_inspect_pick.value.split("|")
        _sz = int(_sz)
        _runs = (act5_df[(act5_df.method_label == _m) & (act5_df.cols == _sz)]
                 .sort_values("world_seed"))
        _ex = _runs.iloc[0]
        _puz = random_world(seed=int(_ex.world_seed), cols=int(_ex.cols), rows=int(_ex.rows),
                            gems=int(_ex.n_gems), density=float(_ex.obstacle_density),
                            locked=int(_ex.n_doors) > 0)
        _env = bp.World(_puz)
        _w = mo.ui.anywidget(_env)
        _plan = [str(a) for a in _ex["plan"]]
        if _plan:
            _env.act(_plan)  # load the MODEL's own plan; the scene's Run button replays it
        _verdict = "✅ solved" if bool(_ex["solved"]) else "❌ failed"
        _out = mo.vstack([
            mo.md(f"### {act5_lab[_m]} · {_sz}×{_sz} · {_verdict}"),
            mo.md(f"The model's plan: {len(_plan)} steps · {int(_ex.wall_time_s)} s · "
                  f"{int(_ex.output_tokens)} tokens · across {len(_runs)} seeds "
                  f"{int(_runs.solved.sum())}/{len(_runs)} solved.  "
                  f"&nbsp; ▶️ Press Run in the scene to watch Mo follow it."),
            _w,
        ])
    _out
    return


@app.cell(hide_code=True)
def act5_table(act5_df):
    _labels = [("plan_nothink", "plan · no-think"), ("plan_think", "plan · think"),
               ("code", "write code"), ("code_fb", "code + verifier")]
    _code_tok = act5_df[act5_df.method_label == "code"].output_tokens.median()
    _rows = []
    for _key, _lbl in _labels:
        _s = act5_df[act5_df.method_label == _key]
        _ratio = _s.output_tokens.median() / _code_tok
        _rows.append(
            f"| **{_lbl}** | {_s.solved.mean():.0%} | {int(_s.wall_time_s.median())} s | "
            f"{int(_s.output_tokens.median()):,} | {_ratio:.1f}× |"
        )
    mo.md(
        "**What each approach costs.** Accuracy next to the compute it took to get there, with "
        "tokens shown relative to `write code`:\n\n"
        "| approach | solved | median time | median tokens | tokens vs. code |\n"
        "|---|:--:|:--:|:--:|:--:|\n" + "\n".join(_rows)
    )
    return


@app.cell(hide_code=True)
def act5_takeaway():
    mo.md(r"""
    ### The verdict

    - **Asking the model to *be* the planner doesn't scale.** No-thinking collapses at 7×7. Thinking holds longer but frays at 16 and 18 (down to 33%) and is dead by 20×20. You pay for it, too: its reasoning climbs roughly linearly with the grid (≈300 s → 1700 s, ~4k → ~15k tokens). Steadily more compute for steadily worse reliability.
    - **Asking the model to *write* a planner does, almost all the way.** `write code` is 100% up to 20×20 at a flat ~50 s / ~800 tokens: a horizontal success line over a gently rising cost curve. At 24×24 it takes its first hit (67%), one world it plans wrong.
    - **And this is where the verifier proves itself useful.** `code + verifier` stays 100% at every size: at 24×24 it catches that bad plan and repairs it, for essentially the same median cost, because the safety net only fires on the rare miss. The LLM-Modulo loop, paying off exactly when raw generation starts to crack.

    That's the LLM-BabyBench thesis, extended: grounded planning is hard for LLMs as direct planners, but reframing it as code generation + verification sidesteps the failure mode, and the verifier is what keeps it standing as the worlds get big.
    """)
    return


@app.cell(hide_code=True)
def act5_frontier():
    mo.callout(
        mo.md(r"""
    **🤖 What about frontier models?**

    The honest answer is: it's complicated. The biggest systems have largely shifted to agentic setups built on _thinking_ models, running far larger and smarter networks than the 31B model here. We couldn't get a true *no-thinking* result out of them at all: handed this exact task they reason by default, and in several cases they wrote and executed Python as part of that thinking, handing back a *provably optimal* plan. Which is precisely Act 4's point: given a runtime, the capable move is to write the planner rather than trace it out in your head.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def fidelity_act5():
    paper_note(
        "What we actually showed: <b>direct planning collapses as the grid grows, while writing a "
        "solver doesn't</b>: a companion finding in our own world, not a reproduction of the "
        "benchmark. The paper's own headline is the Predict↔Plan gap (models <i>describe</i> a "
        "world well but <i>plan</i> in it poorly); we extend its 'planning is the hard part' thesis "
        "with a fix it doesn't test."
    )
    return


@app.cell(hide_code=True)
def references():
    mo.md(r"""
    ## References & further reading

    **The paper this notebook is built on**

    - Choukrani, Malek, Orel, Xie, Iklassov, Takáč, Lahlou. *LLM-BabyBench:
      Understanding and Evaluating Grounded Planning and Reasoning in LLMs.*
      MBZUAI, 2025. [alphaXiv](https://www.alphaxiv.org/abs/2505.12135)

    **"Can LLMs actually plan?": the running debate**

    - Kambhampati. *Can Large Language Models Reason and Plan?* Annals of the
      New York Academy of Sciences, 2024.
      [alphaXiv](https://www.alphaxiv.org/abs/2403.04121)
    - Valmeekam, Marquez, Sreedharan, Kambhampati. *On the Planning Abilities
      of LLMs: A Critical Investigation.* NeurIPS 2023.
      [alphaXiv](https://www.alphaxiv.org/abs/2305.15771)
    - Valmeekam, Marquez, Olmo, Sreedharan, Kambhampati. *PlanBench: An
      Extensible Benchmark for Evaluating LLMs on Planning and Reasoning about
      Change.* NeurIPS 2023. [alphaXiv](https://www.alphaxiv.org/abs/2206.10498)
    - Valmeekam, Stechly, Gundawar, Kambhampati. *A Systematic Evaluation of
      the Planning and Scheduling Abilities of the Reasoning Model o1.* TMLR
      2025. [OpenReview](https://openreview.net/forum?id=FkKBxp0FhR)

    **Pairing LLMs with a sound verifier (the Act 4 idea: LLM-Modulo)**

    - Kambhampati, Valmeekam, Guan, Verma, Stechly, Bhambri, Saldyt, Murthy.
      *Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo
      Frameworks.* ICML 2024.
      [alphaXiv](https://www.alphaxiv.org/abs/2402.01817)
    - Stechly, Valmeekam, Kambhampati. *On the Self-Verification Limitations of
      LLMs on Reasoning and Planning Tasks.* 2024.
      [alphaXiv](https://www.alphaxiv.org/abs/2402.08115)
    - Gundawar, Valmeekam, Verma, Kambhampati. *Robust Planning with Compound
      LLM Architectures: An LLM-Modulo Approach.* 2024.
      [alphaXiv](https://www.alphaxiv.org/abs/2411.14484)

    **Offloading the work to a runtime (why a sandbox helps)**

    - Gao, Madaan, Zhou, Alon, Liu, Yang, Callan, Neubig. *PAL: Program-aided
      Language Models.* ICML 2023.
      [alphaXiv](https://www.alphaxiv.org/abs/2211.10435)
    - Wang, Chen, Yuan, Zhang, Li, Peng, Ji. *Executable Code Actions Elicit
      Better LLM Agents (CodeAct).* ICML 2024.
      [alphaXiv](https://www.alphaxiv.org/abs/2402.01030)
    - Zhang, Kraska, Khattab. *Recursive Language Models.* 2025.
      [alphaXiv](https://www.alphaxiv.org/abs/2512.24601)

    **Built with**

    - **[Wanderland](https://github.com/ktaletsk/wanderland)**, the open-source
      low-poly 3D coding playground (anywidget) that powers Mo's worlds.
      [Read the story](https://taletskiy.com/blogs/wanderland/?utm_source=molab&utm_medium=notebook&utm_campaign=wanderland-launch&utm_content=references) ·
      `pip install wanderland` · MIT.
    """)
    return


if __name__ == "__main__":
    app.run()
