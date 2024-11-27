from django.db import models

class Publisher(models.Model):
    name = models.CharField(max_length=255)
    # Add other relevant fields here

    def _str_(self):
        return self.name

    class Meta:
        db_table = 'Publisher'

class Author(models.Model):
    author_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)

    class Meta:
        db_table = 'Author'

class Category(models.Model):
    catId = models.AutoField(primary_key=True)
    catName = models.CharField(max_length=255)

    class Meta:
        db_table = 'Category'

class Books(models.Model):
    bookId = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=5, decimal_places=2)
    language = models.CharField(max_length=50)
    fileFormat = models.CharField(max_length=50)
    fileSize = models.FloatField()
    publisher = models.CharField(max_length=255)

    class Meta:
        db_table = 'Books'

class BooksAuthor(models.Model):
    bookId = models.ForeignKey(Books, on_delete=models.CASCADE, to_field='bookId')
    authorId = models.ForeignKey(Author, on_delete=models.CASCADE, to_field='author_id')

    class Meta:
        db_table = 'BooksAuthor'
        unique_together = ('bookId', 'authorId')

class BooksCategories(models.Model):
    bookId = models.ForeignKey(Books, on_delete=models.CASCADE, to_field='bookId')
    catId = models.ForeignKey(Category, on_delete=models.CASCADE, to_field='catId')

    class Meta:
        db_table = 'BooksCategories'
        unique_together = ('bookId', 'catId')

class Payment(models.Model):
    payment_id = models.AutoField(primary_key=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateTimeField()
    user_id = models.IntegerField()

    class Meta:
        db_table = 'Payment'

class PaymentDetails(models.Model):
    payment_id = models.ForeignKey(Payment, on_delete=models.CASCADE, to_field='payment_id')
    book_id = models.ForeignKey(Books, on_delete=models.CASCADE, to_field='bookId')

    class Meta:
        db_table = 'PaymentDetails'
        unique_together = ('payment_id', 'book_id')

class Reviews(models.Model):
    reviewId = models.AutoField(primary_key=True)
    rating = models.IntegerField()
    bookId = models.ForeignKey(Books, on_delete=models.CASCADE, to_field='bookId')

    class Meta:
        db_table = 'Reviews'

class UserBooks(models.Model):
    user_id = models.IntegerField()
    book_id = models.ForeignKey(Books, on_delete=models.CASCADE, to_field='bookId')

    class Meta:
        db_table = 'UserBooks'
        unique_together = ('user_id', 'book_id')

class Userr(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_name = models.CharField(max_length=255)
    password = models.CharField(max_length=255)

    class Meta:
        db_table = 'Userr'
