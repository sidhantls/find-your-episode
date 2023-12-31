import streamlit as st
import utils
import pandas as pd
from google.cloud import firestore
import datetime
import json


@st.cache_data()
def get_retrievers():
    data_dir = 'data/office/'
    faiss_path = data_dir + 'faiss_index.index'
    idx_to_metadata_path = data_dir + 'idx_to_metadata.json'
    retriever = utils.Retriever(faiss_path, idx_to_metadata_path)

    data_dir = 'data/parks/'
    faiss_path = data_dir + 'faiss_index.index'
    idx_to_metadata_path = data_dir + 'idx_to_metadata.json'
    retriever2 = utils.Retriever(faiss_path, idx_to_metadata_path)

    return {'Office': retriever, 'Parks': retriever2}


@st.cache_data()
def get_retrievers_keywords():
    data_dir = 'data/office/'
    faiss_path = data_dir + 'faiss_index.index'
    idx_to_metadata_path = data_dir + 'idx_to_metadata.json'
    retriever_keyword1 = utils.KeywordRetriever(idx_to_metadata_path) 

    data_dir = 'data/parks/'
    faiss_path = data_dir + 'faiss_index.index'
    idx_to_metadata_path = data_dir + 'idx_to_metadata.json'
    retriever_keyword2 = utils.KeywordRetriever(idx_to_metadata_path) 

    return {'Office': retriever_keyword1, 'Parks': retriever_keyword2}


def main():
    st.title("TV Show Episode Search")

    # Sidebar navigation
    st.sidebar.image("imgs/dunder_mifflin.png", use_column_width=True)

    st.sidebar.info("You can search for scenes related to specific topics to help find episodes of interest")

    st.sidebar.title("Navigation")
    tabs = ["Office", "Parks"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)
    keyphrase = st.text_input("Enter a keyphrase")
    searchbar = st.button("Search")

    st.sidebar.title("Example Usage")
    st.sidebar.markdown("The Office:\n1. Pam goes to design school\n2. Jim puts Dwight's stapler in Jello\n3. Michael does a Chris Rock impression\n\nParks and Rec:\n1. John Ralphio and Tom\n2. Duke silver playing in concert")

    st.sidebar.title("How it works")

    st.sidebar.info("1. Given a keyphrase, system searches through various scenes of each episode.\n2. Using semantic search it retrieves episodes with most similar scenes matching the query")

    st.sidebar.title("Note for user")
    st.sidebar.markdown(
        "The tool retreives the episode with a corresponding matching scene from the show. \nThe scene description displayed may not always be 100% accurate. Also, items in the searchbox are logged")

    MIN_CHAR, MAX_CHAR = 4, 150
    if searchbar and not keyphrase.strip():
        st.warning("Please enter a keyphrase to search.")
    elif searchbar and len(keyphrase) < MIN_CHAR:
        st.warning('Enter more characters')
    elif searchbar and len(keyphrase) > MAX_CHAR:
        st.warning('Enter less characters')
    elif searchbar and keyphrase.strip():
        retrievers = get_retrievers()
        retrievers_keywords = get_retrievers_keywords()

        if selected_tab not in selected_tab:
            raise ValueError('Selected TVShow not supported')

        keyphrase = keyphrase.strip()
        retriever = retrievers[selected_tab]
        retriever_keyword = retrievers_keywords[selected_tab]

        resp = retriever.get_final_answer(keyphrase)
        resp2 = retriever_keyword.get_final_answer(keyphrase)

        DISPCOLS = ['season', 'episode', 'scene summary', 'score']

        if len(resp2) > 0:
            st.subheader('Matching Episodes')
            df2 = resp2[DISPCOLS]
            st.dataframe(df2)

        elif len(resp['correct']):
            st.subheader('Matching Episodes')
            resp['correct'].drop_duplicates(['season', 'episode'], inplace=True)
            df1 = resp['correct'][DISPCOLS]
            st.dataframe(df1)
        else:
            pass

        if len(resp['similar']) > 0:
            st.subheader('Episodes with Similar Scenes')
            st.dataframe(resp['similar'][DISPCOLS])

        if not len(resp['correct']) and not len(resp['similar']) and not len(resp2):
            st.subheader('No matches')

        if len(keyphrase.strip()) > 3:
            db = get_logger()

            data = {
                "keyword": keyphrase[:80],
                "name": selected_tab,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            db.collection(st.secrets["collection_name"]).add(data)


@st.cache_resource
def get_logger():
    key_dict = json.loads(st.secrets["textkey"])
    db = firestore.Client.from_service_account_info(key_dict)
    return db


if __name__ == "__main__":
    main()
