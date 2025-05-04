import streamlit as st
import base64
import urllib.parse

def generate_social_share_links(image_b64, caption, platform="all"):
    """
    Generate social sharing links for images and captions.
    
    Args:
        image_b64: Base64 encoded image
        caption: Caption text
        platform: Specific platform or "all" for all platforms
        
    Returns:
        Dictionary of platform names and share links/buttons
    """
    # Prepare sharing content
    encoded_caption = urllib.parse.quote(caption)
    
    # Prepare links for different platforms
    links = {}
    
    # Twitter/X share (text only, no direct image sharing via URL)
    twitter_url = f"https://twitter.com/intent/tweet?text={encoded_caption}"
    links["twitter"] = twitter_url
    
    # Facebook share
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u=https://imagecaption.ai&quote={encoded_caption}"
    links["facebook"] = facebook_url
    
    # LinkedIn share
    linkedin_url = f"https://www.linkedin.com/shareArticle?mini=true&url=https://imagecaption.ai&title=Image%20Caption&summary={encoded_caption}"
    links["linkedin"] = linkedin_url
    
    # WhatsApp share
    whatsapp_url = f"https://wa.me/?text={encoded_caption}"
    links["whatsapp"] = whatsapp_url
    
    # Email share
    email_url = f"mailto:?subject=Check%20out%20this%20image%20caption&body={encoded_caption}"
    links["email"] = email_url
    
    if platform != "all":
        # Return only the requested platform
        return {platform: links.get(platform, "")}
    
    return links

def display_social_sharing_buttons(image_b64, caption):
    """
    Display social sharing buttons in the Streamlit app.
    
    Args:
        image_b64: Base64 encoded image
        caption: Caption text
    """
    st.subheader("Share this caption:")
    
    # Get sharing links
    share_links = generate_social_share_links(image_b64, caption)
    
    # Create a row of buttons
    cols = st.columns(5)
    
    # Twitter/X
    with cols[0]:
        st.markdown(f'''
        <a href="{share_links['twitter']}" target="_blank">
            <button style="background-color:#1DA1F2; color:white; border:none; border-radius:5px; padding:10px; font-size:16px; width:100%;">
                Twitter
            </button>
        </a>
        ''', unsafe_allow_html=True)
    
    # Facebook
    with cols[1]:
        st.markdown(f'''
        <a href="{share_links['facebook']}" target="_blank">
            <button style="background-color:#4267B2; color:white; border:none; border-radius:5px; padding:10px; font-size:16px; width:100%;">
                Facebook
            </button>
        </a>
        ''', unsafe_allow_html=True)
    
    # LinkedIn
    with cols[2]:
        st.markdown(f'''
        <a href="{share_links['linkedin']}" target="_blank">
            <button style="background-color:#0077B5; color:white; border:none; border-radius:5px; padding:10px; font-size:16px; width:100%;">
                LinkedIn
            </button>
        </a>
        ''', unsafe_allow_html=True)
    
    # WhatsApp
    with cols[3]:
        st.markdown(f'''
        <a href="{share_links['whatsapp']}" target="_blank">
            <button style="background-color:#25D366; color:white; border:none; border-radius:5px; padding:10px; font-size:16px; width:100%;">
                WhatsApp
            </button>
        </a>
        ''', unsafe_allow_html=True)
    
    # Email
    with cols[4]:
        st.markdown(f'''
        <a href="{share_links['email']}" target="_blank">
            <button style="background-color:#808080; color:white; border:none; border-radius:5px; padding:10px; font-size:16px; width:100%;">
                Email
            </button>
        </a>
        ''', unsafe_allow_html=True)
    
    # Clipboard copy option
    st.markdown("---")
    
    # Text area for copying the caption
    st.text_area("Copy caption text:", value=caption, height=100)
    
    st.info("Click on any button above to share this caption on your favorite platform.")

def generate_download_link(img, caption, filename="captioned_image.jpg"):
    """
    Generate a download link for the image with caption text overlay.
    
    Args:
        img: PIL Image object
        caption: Caption text
        filename: Filename for download
        
    Returns:
        HTML link for downloading the image
    """
    # Prepare image for download (convert to bytes)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create download link using HTML
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">Download Image</a>'
    
    return href