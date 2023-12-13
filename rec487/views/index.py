"""
Insta485 index (main) view.

URLs include:
/
"""
import flask
import arrow
from flask import request
import rec487
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model.emotions_bayes import EmotionsNaiveBayes
from emotions_dataset_tfidf import EmotionsDatasetTFIDF
from spotify_dataset import SpotifyDataset

@rec487.app.route("/")
def show_index():
    context = {"active":False}
    return flask.render_template("index.html", **context)


@rec487.app.route("/recommender/", methods=["POST"])
def recommend_songs():

    context = {"active":True}
    print(request.form)
    song = request.form["song"]
    # model = request.form["model"]
    k_recs = request.form["k"]

    context["song"] = song
    context["num_recs"] = k_recs
    context["sim_songs"] = []
    spotify = SpotifyDataset('data/spotify_millsongdata.csv')
    print("spotify Data Loaded")

    with open('bayes_data.pkl', 'rb') as inp:
        label_predictor = pickle.load(inp)

    song_index = spotify.get_index_from_title(song)
    if song_index:
        context["available"] = True
        neighbor_indices = label_predictor.get_recommendations(song_index, int(k_recs))
        sim_songs = []
        print('Similar Songs!')
        for i, idx in enumerate(neighbor_indices):
            sim_songs.append({"num":i + 1, "title":spotify.get_title(idx)})
            print(f'{i}:', spotify.get_title(idx))
        print(sim_songs)
        context["sim_songs"] = sim_songs
    else:
        context["available"] = False
    return flask.render_template("index.html", **context)
