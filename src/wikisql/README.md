### Generating WikiSQL Dataset for MMTABQA

To generate the WikiSQL dataset for MMTABQA, refer to the notebook **`WIKISQL-MM.ipynb`**. In addition, you will need to utilize several Python scripts from the `utils` directory for various tasks:

1. **Get Revision IDs**:  
   Use **`revision_id.py`** to fetch revision IDs for Wikipedia pages.

2. **HTML Page Scraping**:  
   Use **`html_scraping.py`** to scrape the required HTML pages.

3. **Table Extraction**:  
   Extract tables from the scraped HTML pages with **`download_all_tables.py`**.

4. **Get Similar Tables and Wikipedia Links**:  
   Use **`get_similar_tables.py`** to find similar tables on the HTML pages and retrieve links to Wikipedia pages for different entities.

5. **Get Image Links from Wikipedia Pages**:  
   Extract links to images from Wikipedia pages using **`links_to_image_pipeline.py`**.

6. **Download Images**:  
   Download images from the extracted links with **`download_images.py`**.

7. **Generate Categories for Entities**:  
   Categorize entities based on the annotations with **`parse_categorywise_entity_annotations.ipynb`**.

8. **Categorize Images**:  
   Categorize images based on the entity categories

9. **Segregate Images**:  
   Use **`segregate.py`** to segregate images based on the entity categories.

10. **Create Visual Questions**:  
   Generate visual questions with **`visual_qs_gen_pipeline.ipynb`**.

Each of these steps is essential for building the dataset as described in the notebook.
