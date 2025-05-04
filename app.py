import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import time
import base64
import os
from model import ImageCaptioningModel
from utils import preprocess_image, get_device
import database as db
import auth

# Page configuration
st.set_page_config(
    page_title="Image Captioning AI",
    page_icon="🖼️",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'view_history' not in st.session_state:
    st.session_state.view_history = False
if 'selected_image_id' not in st.session_state:
    st.session_state.selected_image_id = None

# Initialize authentication state
auth.init_session_state()

# Function to load model
@st.cache_resource
def load_model():
    device = get_device()
    model = ImageCaptioningModel(device=device)
    return model

# Create sidebar for navigation and authentication status
with st.sidebar:
    st.title("🖼️ Image Captioning AI")
    
    # Display user info or login link
    if auth.is_authenticated():
        user = auth.get_current_user()
        st.success(f"Logged in as: {user['username']}")
        
        # Navigation links
        st.subheader("Navigation")
        nav_option = st.radio(
            "Go to:",
            ["Home", "My Profile", "My Captions"]
        )
        
        if st.button("Logout"):
            auth.logout_user()
            st.rerun()
    else:
        st.warning("You are not logged in")
        st.info("Login to save and manage your captioned images")
        
        # Navigation for guests
        st.subheader("Navigation")
        nav_option = st.radio(
            "Go to:",
            ["Home", "Login/Register"]
        )
    
    # App information
    st.markdown("---")
    st.subheader("About")
    st.info("""
    Image Captioning AI combines computer vision and natural language processing to generate 
    descriptive captions for your images.
    """)

# Determine which page to show based on navigation
if auth.is_authenticated():
    # Authenticated user navigation
    if nav_option == "Home":
        show_page = "home"
    elif nav_option == "My Profile":
        show_page = "profile"
    elif nav_option == "My Captions":
        show_page = "my_captions"
else:
    # Guest navigation
    if nav_option == "Home":
        show_page = "home"
    elif nav_option == "Login/Register":
        show_page = "auth"

# Home page with image captioning functionality
if show_page == "home":
    st.title("🖼️ Image Captioning AI")
    
    # Navigation tabs for the home page
    tab1, tab2, tab3 = st.tabs(["📸 Generate Captions", "📚 Caption History", "🔍 Compare Images"])
    
    with tab1:
        st.markdown("""
        Upload an image and our AI will generate a descriptive caption for it.
        This app uses computer vision to extract features from images and natural language processing to generate captions.
        """)
    
        # Model loading section
        with st.spinner("Loading model... This may take a minute."):
            if not st.session_state.model_loaded:
                try:
                    st.session_state.model = load_model()
                    st.session_state.model_loaded = True
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()
    
        # Image upload section
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file)
                # Create two columns for image and results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    # Generate caption for the image
                    with st.spinner("Generating caption..."):
                        try:
                            # Preprocess the image for the model
                            processed_image = preprocess_image(image)
                            
                            if processed_image is None:
                                st.error("Error preprocessing the image. Please try a different image.")
                                st.stop()
                            
                            # Generate caption
                            start_time = time.time()
                            caption = st.session_state.model.generate_caption(processed_image)
                            end_time = time.time()
                            processing_time = end_time - start_time
                            
                            if caption is None or caption == "I can't describe this image. Please try another one.":
                                st.error("Could not generate a meaningful caption. Please try another image.")
                                st.stop()
                            
                            # Save to database (with user_id if authenticated)
                            try:
                                user_id = None
                                if auth.is_authenticated():
                                    user_id = auth.get_current_user()['id']
                                    
                                db.save_image_caption(
                                    image=image, 
                                    caption=caption, 
                                    filename=uploaded_file.name,
                                    processing_time=processing_time,
                                    user_id=user_id,
                                    is_public=True  # Default to public
                                )
                                # Display results
                                st.subheader("Generated Caption:")
                                st.write(f"**{caption}**")
                                st.write(f"Processing time: {processing_time:.2f} seconds")
                                st.success("Image and caption saved to database!")
                            except Exception as e:
                                st.subheader("Generated Caption:")
                                st.write(f"**{caption}**")
                                st.write(f"Processing time: {processing_time:.2f} seconds")
                                st.warning(f"Caption generated but not saved to database: {str(e)}")
                        except Exception as e:
                            st.error(f"Error generating caption: {str(e)}")
                            st.text("Please try a different image or try again later.")
                        
                        # Privacy options for authenticated users
                        if auth.is_authenticated():
                            st.markdown("---")
                            st.subheader("Privacy Settings:")
                            st.info("Your caption is currently public (visible to all users)")
                            if st.button("Make Private"):
                                # Get the latest caption for this user
                                user_captions = db.get_all_captions(limit=1, user_id=user_id)
                                if user_captions:
                                    latest_caption_id = user_captions[0]['id']
                                    db.update_caption_privacy(latest_caption_id, user_id, False)
                                    st.success("Caption privacy updated to private")
                                    st.rerun()
                
                # Image enhancement section
                st.markdown("---")
                st.subheader("Image Enhancement")
                
                # Import image_effects module
                import image_effects
                
                # Get available filters
                available_filters = image_effects.get_available_filters()
                
                # Create filter selector
                selected_filter = st.selectbox(
                    "Apply a filter to your image:",
                    available_filters,
                    index=0  # Default to 'original'
                )
                
                # Apply selected filter
                if selected_filter != "original":
                    filtered_image = image_effects.apply_filter(image, selected_filter)
                    
                    # Show filtered image
                    st.image(filtered_image, caption=f"{selected_filter.title()} Filter Applied", use_container_width=True)
                    
                    # Option to save filtered image
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Generate new caption for filtered image
                        if st.button("Re-caption Filtered Image"):
                            with st.spinner("Generating new caption..."):
                                try:
                                    # Preprocess the filtered image
                                    processed_filtered = preprocess_image(filtered_image)
                                    
                                    if processed_filtered is None:
                                        st.error("Error preprocessing the filtered image.")
                                        st.stop()
                                    
                                    # Generate caption
                                    start_time = time.time()
                                    new_caption = st.session_state.model.generate_caption(processed_filtered)
                                    end_time = time.time()
                                    new_processing_time = end_time - start_time
                                    
                                    if new_caption is None or new_caption == "I can't describe this image. Please try another one.":
                                        st.error("Could not generate a meaningful caption for the filtered image.")
                                        st.stop()
                                    
                                    # Save to database with user_id if authenticated
                                    try:
                                        user_id = None
                                        if auth.is_authenticated():
                                            user_id = auth.get_current_user()['id']
                                            
                                        db.save_image_caption(
                                            image=filtered_image, 
                                            caption=new_caption, 
                                            filename=f"{selected_filter}_{uploaded_file.name}" if uploaded_file.name else f"{selected_filter}_image.jpg",
                                            processing_time=new_processing_time,
                                            user_id=user_id
                                        )
                                        st.success(f"New caption for filtered image: **{new_caption}**")
                                        st.success("Filtered image and caption saved to database!")
                                    except Exception as e:
                                        st.success(f"New caption for filtered image: **{new_caption}**")
                                        st.warning(f"Caption generated but not saved to database: {str(e)}")
                                except Exception as e:
                                    st.error(f"Error generating caption for filtered image: {str(e)}")
                                    st.text("Please try a different filter or image.")
                    
                    with col2:
                        # Save filtered image
                        buffered = io.BytesIO()
                        filtered_image.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        href = f'<a href="data:file/jpg;base64,{img_str}" download="{selected_filter}_{uploaded_file.name if uploaded_file.name else "image.jpg"}">Download Filtered Image</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                # Image analysis section
                st.markdown("---")
                st.subheader("Image Analysis")
                
                # Import image_analysis module
                import image_analysis
                
                # Create expander for image analysis
                with st.expander("View Image Analysis"):
                    # Analyze the image
                    analysis = image_analysis.analyze_image(image)
                    quality = image_analysis.estimate_image_quality(image)
                    
                    # Display analysis in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Image Details:**")
                        st.write(f"Dimensions: {analysis['dimensions']}")
                        st.write(f"Estimated Size: {analysis['file_size']}")
                        st.write(f"Format: {analysis['format']}")
                        st.write(f"Mode: {analysis['mode']}")
                        
                        st.write("**Color Information:**")
                        st.write(f"Dominant Color: ")
                        st.markdown(f"""
                        <div style="background-color:{analysis['dominant_color']}; 
                                    width:50px; 
                                    height:20px; 
                                    display:inline-block; 
                                    margin-right:10px; 
                                    border:1px solid black;">
                        </div> {analysis['dominant_color']}
                        """, unsafe_allow_html=True)
                        
                        st.write(f"Brightness: {analysis['brightness']}")
                        st.write(f"Contrast: {analysis['contrast']}")
                    
                    with col2:
                        st.write("**Image Quality Assessment:**")
                        st.write(f"Resolution Quality: {quality['resolution_quality']}")
                        st.write(f"Contrast Quality: {quality['contrast_quality']}")
                        st.write(f"Overall Quality: {quality['overall_quality']} ({quality['overall_score']})")
                        
                        st.write("**Channel Distribution:**")
                        st.write(f"Red: {analysis['color_balance']['red']}")
                        st.write(f"Green: {analysis['color_balance']['green']}")
                        st.write(f"Blue: {analysis['color_balance']['blue']}")
                    
                    # Display color histogram
                    st.write("**Color Histograms:**")
                    histogram_data = image_analysis.generate_color_histogram(image)
                    st.image(f"data:image/png;base64,{histogram_data}")
                
                # Social sharing section
                st.markdown("---")
                st.subheader("Share Your Caption")
                
                # Import sharing module
                import sharing
                
                # Convert image to base64 for sharing
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Display sharing buttons
                sharing.display_social_sharing_buttons(img_str, caption)
                
                # Additional information
                st.markdown("---")
                st.subheader("How it works:")
                st.markdown("""
                1. The system extracts features from the image using a pre-trained ResNet50 model
                2. These features are then processed by a decoder network that generates a caption
                3. The caption is generated word by word, with each word conditioned on the image features and previous words
                4. The image and caption are stored in the database for future reference
                5. You can enhance your image with filters and analyze its properties
                6. Share your captioned images on social media platforms
                """)
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.error("Full error details:")
                st.exception(e)
        else:
            # Show placeholder or example when no image is uploaded
            st.info("Please upload an image to generate a caption.")
            
            # Information about the application
            st.markdown("---")
            st.subheader("About this Image Captioning System")
            st.markdown("""
            This application uses deep learning techniques to automatically generate descriptive captions for images:
            
            - **Computer Vision**: A pre-trained ResNet50 CNN extracts visual features from the uploaded image
            - **Natural Language Processing**: An LSTM-based decoder converts visual features into natural language
            - **End-to-End Learning**: The system was trained on image-caption pairs to learn the relationship between visual content and descriptive language
            - **Database Storage**: All generated captions and images are stored for future reference
            - **User Accounts**: Create an account to manage your captioned images and privacy settings
            
            Supported image formats: JPEG, PNG
            """)
    
    with tab2:
        st.subheader("Previously Captioned Images")
        
        # Get image captions from database (with user_id if authenticated)
        user_id = None
        if auth.is_authenticated():
            user_id = auth.get_current_user()['id']
            
        captions = db.get_all_captions(limit=20, user_id=user_id)
        
        if not captions:
            st.info("No captioned images found in the database. Upload an image in the 'Generate Captions' tab to get started.")
        else:
            # Display image thumbnails in a grid
            st.write(f"Showing {len(captions)} most recent captioned images:")
            
            # Create rows with 4 columns each
            cols_per_row = 4
            for i in range(0, len(captions), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(captions):
                        caption_data = captions[i + j]
                        with cols[j]:
                            # Display thumbnail
                            st.image(
                                f"data:image/jpeg;base64,{caption_data['thumbnail']}", 
                                caption=caption_data['caption'][:20] + "..." if len(caption_data['caption']) > 20 else caption_data['caption'],
                                use_container_width=True
                            )
                            
                            # Display username if available
                            if caption_data.get('username'):
                                st.caption(f"By: {caption_data['username']}")
                                
                            # Button to view full details
                            if st.button(f"View details", key=f"btn_{caption_data['id']}"):
                                st.session_state.selected_image_id = caption_data['id']
        
        # Display selected image details if any
    
    with tab3:
        st.subheader("Compare Two Images")
        st.markdown("""
        Upload two images to compare their features and see how similarly they are captioned by our AI.
        This tool helps you understand the differences and similarities between images.
        """)
        
        # Initialize session state for comparison
        if 'comparison_image1' not in st.session_state:
            st.session_state.comparison_image1 = None
        if 'comparison_image2' not in st.session_state:
            st.session_state.comparison_image2 = None
        if 'comparison_result' not in st.session_state:
            st.session_state.comparison_result = None
        
        # Import image_comparison module
        import image_comparison
        
        # Create two columns for uploading images
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Image 1:**")
            upload1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="upload1")
            
            if upload1:
                try:
                    image1 = Image.open(upload1)
                    st.image(image1, caption="Image 1", use_container_width=True)
                    st.session_state.comparison_image1 = image1
                except Exception as e:
                    st.error(f"Error loading image 1: {str(e)}")
        
        with col2:
            st.write("**Image 2:**")
            upload2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="upload2")
            
            if upload2:
                try:
                    image2 = Image.open(upload2)
                    st.image(image2, caption="Image 2", use_container_width=True)
                    st.session_state.comparison_image2 = image2
                except Exception as e:
                    st.error(f"Error loading image 2: {str(e)}")
        
        # Comparison section (only if both images are uploaded)
        if st.session_state.comparison_image1 is not None and st.session_state.comparison_image2 is not None:
            st.markdown("---")
            
            if st.button("Compare Images"):
                with st.spinner("Analyzing and comparing images..."):
                    # Compare images
                    comparison = image_comparison.compare_images(
                        st.session_state.comparison_image1, 
                        st.session_state.comparison_image2
                    )
                    
                    # Create difference image
                    diff_img = image_comparison.create_difference_image(
                        st.session_state.comparison_image1, 
                        st.session_state.comparison_image2
                    )
                    
                    # Create side-by-side comparison
                    side_by_side = image_comparison.create_side_by_side_comparison(
                        st.session_state.comparison_image1, 
                        st.session_state.comparison_image2, 
                        captions=("Image 1", "Image 2")
                    )
                    
                    # Generate captions for both images
                    if st.session_state.model_loaded:
                        processed_img1 = preprocess_image(st.session_state.comparison_image1)
                        processed_img2 = preprocess_image(st.session_state.comparison_image2)
                        
                        caption1 = st.session_state.model.generate_caption(processed_img1)
                        caption2 = st.session_state.model.generate_caption(processed_img2)
                    else:
                        caption1 = "Model not loaded. Cannot generate caption."
                        caption2 = "Model not loaded. Cannot generate caption."
                    
                    # Store result in session state
                    st.session_state.comparison_result = {
                        "comparison": comparison,
                        "diff_img": diff_img,
                        "side_by_side": side_by_side,
                        "caption1": caption1,
                        "caption2": caption2
                    }
                
                st.success("Comparison complete!")
            
            # Display comparison results if available
            if st.session_state.comparison_result:
                result = st.session_state.comparison_result
                
                # Display side-by-side comparison
                st.subheader("Side-by-Side Comparison")
                
                # Convert to bytes for display
                buffered = io.BytesIO()
                result["side_by_side"].save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                st.image(f"data:image/jpeg;base64,{img_str}", use_container_width=True)
                
                # Display difference image
                st.subheader("Difference Visualization")
                st.write("This image highlights the differences between the two images. Brighter areas indicate larger differences.")
                
                # Convert to bytes for display
                buffered = io.BytesIO()
                result["diff_img"].save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                st.image(f"data:image/jpeg;base64,{img_str}", use_container_width=True)
                
                # Display numerical comparison
                st.subheader("Image Similarity Metrics")
                
                metrics = result["comparison"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Dimensions:** {metrics['dimensions']}")
                    st.write(f"**Similarity Percentage:** {metrics['similarity_percentage']}")
                    st.write(f"**Mean Squared Error:** {metrics['mse']}")
                
                with col2:
                    st.write(f"**Structural Similarity:** {metrics['ssim']}")
                    st.write(f"**Histogram Correlation:** {metrics['histogram_correlation']}")
                
                # Display captions
                st.subheader("Generated Captions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Image 1 Caption:**")
                    st.write(f"*{result['caption1']}*")
                
                with col2:
                    st.write("**Image 2 Caption:**")
                    st.write(f"*{result['caption2']}*")
                
                # Help text
                st.info("""
                **How to interpret these metrics:**
                - **Similarity Percentage:** Higher values indicate more similar images
                - **Structural Similarity:** Values closer to 1 indicate similar structure
                - **Mean Squared Error:** Lower values indicate more similar pixel values
                - **Histogram Correlation:** Higher values indicate similar color distributions
                """)
        
        # Display selected image details if any
        if st.session_state.selected_image_id:
            st.markdown("---")
            st.subheader("Image Details")
            
            # Get the selected image details
            current_user_id = None
            if auth.is_authenticated():
                current_user_id = auth.get_current_user()['id']
                
            image_data = db.get_caption_by_id(st.session_state.selected_image_id, user_id=current_user_id)
            
            if image_data:
                # Create two columns for image and details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        f"data:image/jpeg;base64,{image_data['image']}", 
                        caption=f"Filename: {image_data['filename'] or 'Unknown'}", 
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("Generated Caption:")
                    st.write(f"**{image_data['caption']}**")
                    
                    # Show username if available
                    if image_data.get('username'):
                        st.write(f"By: **{image_data['username']}**")
                        
                    st.write(f"Generated on: {image_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"Processing time: {image_data['processing_time']:.2f} seconds")
                    st.write(f"Privacy: **{'Public' if image_data['is_public'] else 'Private'}**")
                    
                    # Show privacy toggle and delete button if user owns this caption
                    if auth.is_authenticated() and image_data['user_id'] == current_user_id:
                        # Privacy toggle
                        if st.button(
                            "Make Private" if image_data['is_public'] else "Make Public", 
                            key=f"privacy_{image_data['id']}"
                        ):
                            if db.update_caption_privacy(image_data['id'], current_user_id, not image_data['is_public']):
                                st.success(f"Caption privacy updated to {'private' if image_data['is_public'] else 'public'}")
                                st.rerun()
                            else:
                                st.error("Failed to update privacy setting")
                        
                        # Delete button
                        if st.button("Delete this caption", key=f"del_{image_data['id']}"):
                            if db.delete_caption(image_data['id'], user_id=current_user_id):
                                st.success("Caption deleted successfully!")
                                st.session_state.selected_image_id = None
                                st.rerun()
                            else:
                                st.error("Failed to delete caption")
                    # For non-owners, only show delete button if they're an unauthenticated user looking at an unclaimed caption
                    elif not auth.is_authenticated() and image_data['user_id'] is None:
                        if st.button("Delete this record", key=f"del_{image_data['id']}"):
                            if db.delete_caption(image_data['id']):
                                st.success("Record deleted successfully!")
                                st.session_state.selected_image_id = None
                                st.rerun()
                            else:
                                st.error("Failed to delete record")
            else:
                st.error("Image not found. It may have been deleted or you don't have permission to view it.")
                st.session_state.selected_image_id = None

# Authentication page
elif show_page == "auth":
    auth.display_auth_page()

# User profile page
elif show_page == "profile":
    auth.display_user_profile()

# My captions page (shows only the user's captions)
elif show_page == "my_captions":
    # Apply the auth_required decorator to protect this page
    @auth.auth_required
    def show_my_captions():
        st.title("My Captioned Images")
        
        user = auth.get_current_user()
        user_id = user['id']
        
        # Get only this user's captions
        user_captions = db.get_all_captions(limit=50, user_id=user_id)
        
        if not user_captions:
            st.info("You haven't captioned any images yet. Go to the Home page to get started!")
            return
            
        # Create filter for public/private
        filter_option = st.radio("Show:", ["All Captions", "Public Only", "Private Only"], horizontal=True)
        
        filtered_captions = []
        if filter_option == "All Captions":
            filtered_captions = user_captions
        elif filter_option == "Public Only":
            filtered_captions = [c for c in user_captions if c.get('is_public')]
        elif filter_option == "Private Only":
            filtered_captions = [c for c in user_captions if not c.get('is_public')]
        
        # Display count of captions
        st.write(f"Showing {len(filtered_captions)} images:")
        
        # Create rows with 4 columns each
        cols_per_row = 4
        for i in range(0, len(filtered_captions), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(filtered_captions):
                    caption_data = filtered_captions[i + j]
                    with cols[j]:
                        # Display thumbnail with privacy indicator
                        privacy_icon = "🌐" if caption_data['is_public'] else "🔒"
                        st.image(
                            f"data:image/jpeg;base64,{caption_data['thumbnail']}", 
                            caption=f"{privacy_icon} {caption_data['caption'][:20] + '...' if len(caption_data['caption']) > 20 else caption_data['caption']}",
                            use_container_width=True
                        )
                        
                        # Display timestamp
                        st.caption(f"{caption_data['timestamp'].strftime('%Y-%m-%d')}")
                        
                        # Buttons for actions
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("View", key=f"view_{caption_data['id']}"):
                                st.session_state.selected_image_id = caption_data['id']
                                st.rerun()
                        with col2:
                            privacy_label = "Private" if caption_data['is_public'] else "Public"
                            if st.button(f"Make {privacy_label}", key=f"privacy_{caption_data['id']}"):
                                if db.update_caption_privacy(caption_data['id'], user_id, not caption_data['is_public']):
                                    st.success(f"Made {privacy_label.lower()}")
                                    st.rerun()
        
        # Display selected image if any
        if st.session_state.selected_image_id:
            st.markdown("---")
            st.subheader("Image Details")
            
            image_data = db.get_caption_by_id(st.session_state.selected_image_id, user_id=user_id)
            
            if image_data:
                # Create two columns for image and details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        f"data:image/jpeg;base64,{image_data['image']}", 
                        caption=f"Filename: {image_data['filename'] or 'Unknown'}", 
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("Generated Caption:")
                    st.write(f"**{image_data['caption']}**")
                    st.write(f"Generated on: {image_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"Processing time: {image_data['processing_time']:.2f} seconds")
                    
                    # Privacy status and toggle
                    st.markdown("---")
                    privacy_status = "Public (visible to everyone)" if image_data['is_public'] else "Private (visible only to you)"
                    st.write(f"**Privacy Status:** {privacy_status}")
                    
                    if st.button(
                        "Make Private" if image_data['is_public'] else "Make Public", 
                        key=f"toggle_privacy_{image_data['id']}"
                    ):
                        if db.update_caption_privacy(image_data['id'], user_id, not image_data['is_public']):
                            st.success(f"Privacy setting updated")
                            st.rerun()
                        else:
                            st.error("Failed to update privacy setting")
                    
                    # Delete button
                    st.markdown("---")
                    if st.button("Delete this caption", key=f"delete_{image_data['id']}"):
                        if db.delete_caption(image_data['id'], user_id=user_id):
                            st.success("Caption deleted successfully!")
                            st.session_state.selected_image_id = None
                            st.rerun()
                        else:
                            st.error("Failed to delete caption")
            else:
                st.error("Image not found. It may have been deleted.")
                st.session_state.selected_image_id = None
    
    # Call the protected function
    show_my_captions()

# Footer
st.markdown("---")
st.caption("Image Captioning AI • Built with Streamlit, PyTorch, and SQLAlchemy")
