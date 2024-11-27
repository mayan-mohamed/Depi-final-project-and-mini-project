from django.contrib import admin
from .models import Author, Category, Books, BooksAuthor, BooksCategories, Payment, PaymentDetails, Reviews, UserBooks, Userr

# Registering the models
admin.site.register(Author)
admin.site.register(Category)
admin.site.register(Books)
admin.site.register(BooksAuthor)
admin.site.register(BooksCategories)
admin.site.register(Payment)
admin.site.register(PaymentDetails)
admin.site.register(Reviews)
admin.site.register(UserBooks)
admin.site.register(Userr)
