"""
Developer: Tsering Dhargyal
Project : Intelligent recommendation system
Date : 18/11/2022, Saturday
"""

#function to load the datasets 

def loadDataSet():
    booksDict={}
    try:
        with open("/Users/dhargyal/Desktop/mscBigData/PCP/Assignment/Books/Books.csv", "r", encoding="latin-1") as books:
            headerInfo = books.readline()
            for book in books:
                isbn = book.split(";")[0].strip("=")
                booksDict[isbn]= isbn.rstrip()
                
    except IOError as ioerror:
        print("File error",ioerror)

    userDict = {}
    try:
        with open("/Users/dhargyal/Desktop/mscBigData/PCP/Assignment/Books/Users.csv","r", encoding="latin-1") as users:
            headerInfo = users.readline()
            for user in users:
                userId = user.split(";")[0].strip("=")
                userDict[userId]=user
                  

    except IOError as ioerror:
        print("File error",ioerror)

    bookRatingDict = {}
    try:
        with open("/Users/dhargyal/Desktop/mscBigData/PCP/Assignment/Books/Book-Ratings.csv","r", encoding="latin-1") as ratings:
            headerInfo = ratings.readline()
            for rating in ratings:
                bookRating = rating.split(";")[2].strip("=")
                bookRatingDict[bookRating]=rating
            
                
               

    except IOError as ioerror:
        print("File error",ioerror)

    user_preference ={}
    try:
       user_rating = bookRatingDict | userDict
       book_user_rating = booksDict | user_rating
       #book_user_rating = book_user_rating[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
       
    except IOError as ioerror:
        print("File error",ioerror)



loadDataSet()