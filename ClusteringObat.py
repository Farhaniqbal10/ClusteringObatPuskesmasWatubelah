from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from io import BytesIO
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from num2words import num2words
from functools import reduce

st.set_page_config(page_title="Data Mining Obat")

class MainClass():

    def __init__(self):
        # Inisiasi objek
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.dbi = Dbi()
        self.clustering = Clustering()
        self.rekomendasi = Rekomendasi() 

    # Fungsi judul halaman
    def judul_halaman(self, header, subheader):
        nama_app = "Aplikasi Data Mining Obat"
        st.title(nama_app)
        st.header(header)
        st.subheader(subheader)
    
    # Fungsi menu sidebar
    def sidebar_menu(self):
        with st.sidebar:
            selected = option_menu('Menu', ['Data', 'Pre Processing dan Transformation', 'DBI', 'Clustering', 'Rekomendasi'], default_index=0)
            
        if selected == 'Data':
            self.data.menu_data()

        if selected == 'Pre Processing dan Transformation':
            self.preprocessing.menu_preprocessing()

        if selected == 'DBI':
            self.dbi.menu_dbi()

        if selected == 'Clustering':
            self.clustering.menu_clustering()
                    
        if selected == 'Rekomendasi':
            self.rekomendasi.menu_rekomendasi()

# class Data(MainClass):

#     def __init__(self):
#         # Membuat state untuk menampung dataframe
#         self.state = st.session_state.setdefault('state', {})
#         if 'dataobat' not in self.state:
#             self.state['dataobat'] = pd.DataFrame()
#         if 'datarm' not in self.state:
#             self.state['datarm'] = pd.DataFrame()

#     def upload_dataobat(self):
#         try:
#             uploaded_file1 = st.file_uploader("Upload Data Obat", type=["xlsx"], key="obat")
#             if uploaded_file1 is not None:
#                 self.state['dataobat'] = pd.DataFrame()
#                 fobat = pd.ExcelFile(uploaded_file1)

#                 # Membaca file excel dari banyak sheet
#                 list_of_dfs_obat = []
#                 for sheet in fobat.sheet_names:
#                     # Parse data from each worksheet as a Pandas DataFrame
#                     dfobat = fobat.parse(sheet, header=0)
#                     # And append it to the list
#                     list_of_dfs_obat.append(dfobat)

#                 # Combine all DataFrames into one, 
#                 dataobat = pd.concat(list_of_dfs_obat, ignore_index=True).drop_duplicates()

#                 self.state['dataobat'] = dataobat
#                 st.success("Data obat uploaded successfully!")

#         except (TypeError, IndexError, KeyError) as e:
#             st.error(f"Data yang diupload tidak sesuai: {e}")

#     def display_data_obat(self):
#         if 'dataobat' in self.state:
#             if not self.state['dataobat'].empty:
#                 st.subheader("Data Obat")
#                 st.write(self.state['dataobat'])
#             else:
#                 st.write("No data available")
#         else:
#             st.write("No data uploaded yet")

#     def upload_datarm(self):
#         try:
#             uploaded_file2 = st.file_uploader("Upload Data Rekam Medis", type=["xlsx"], key="rekammedis")
#             if uploaded_file2 is not None:
#                 self.state['datarm'] = pd.DataFrame()
#                 frekammedis = pd.ExcelFile(uploaded_file2)

#                 # Membaca file excel dari banyak sheet
#                 list_of_dfs_rekammedis = []
#                 for sheet in frekammedis.sheet_names:
#                     # Parse data from each worksheet as a Pandas DataFrame
#                     dfrekammedis = frekammedis.parse(sheet)
#                     # And append it to the list
#                     list_of_dfs_rekammedis.append(dfrekammedis)

#                 # Combine all DataFrames into one
#                 datarekammedis = pd.concat(list_of_dfs_rekammedis, ignore_index=True).drop_duplicates()

#                 self.state['datarm'] = datarekammedis
#                 st.success("Data rekam medis uploaded successfully!")

#         except (TypeError, IndexError, KeyError) as e:
#             st.error(f"Data yang diupload tidak sesuai: {e}")

#     def display_datarm(self):
#         if 'datarm' in self.state:
#             if not self.state['datarm'].empty:
#                 st.subheader("Data Rekam Medis")
#                 st.write(self.state['datarm'])
#             else:
#                 st.write("No data available")
#         else:
#             st.write("No data uploaded yet")

#     def tampil_dataobat(self):
#         if not self.state['dataobat'].empty:
#             st.dataframe(self.state['dataobat'])

#     def tampil_datarm(self):
#         if not self.state['datarm'].empty:
#             st.dataframe(self.state['datarm'])

#     def menu_data(self):
#         self.judul_halaman('Data', 'Import Dataset')
#         self.upload_dataobat()
#         self.tampil_dataobat()
#         self.upload_datarm()
#         self.tampil_datarm()

class Data(MainClass):

    def __init__(self):
        self.state = st.session_state.setdefault('state', {})
        if 'dataobat' not in self.state:
            self.state['dataobat'] = pd.DataFrame()
        if 'datarm' not in self.state:
            self.state['datarm'] = pd.DataFrame()

    def check_required_columns(self, df, required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def upload_dataobat(self):
        try:
            uploaded_file1 = st.file_uploader("Upload Data Obat", type=["xlsx"], key="obat")
            if uploaded_file1 is not None:
                self.state['dataobat'] = pd.DataFrame()
                fobat = pd.ExcelFile(uploaded_file1)

                list_of_dfs_obat = []
                for sheet in fobat.sheet_names:
                    dfobat = fobat.parse(sheet, header=0)
                    
                    # Cek apakah kolom yang dibutuhkan ada di dalam data
                    self.check_required_columns(dfobat, ['STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR'])
                    
                    list_of_dfs_obat.append(dfobat)

                dataobat = pd.concat(list_of_dfs_obat, ignore_index=True).drop_duplicates()
                self.state['dataobat'] = dataobat
                st.success("Data obat uploaded successfully!")

        except ValueError as e:
            st.error(f"Error in data obat: {e}")
        except (TypeError, IndexError, KeyError) as e:
            st.error(f"Data yang diupload tidak sesuai: {e}")

    def upload_datarm(self):
        try:
            uploaded_file2 = st.file_uploader("Upload Data Rekam Medis", type=["xlsx"], key="rekammedis")
            if uploaded_file2 is not None:
                self.state['datarm'] = pd.DataFrame()
                frekammedis = pd.ExcelFile(uploaded_file2)

                list_of_dfs_rekammedis = []
                for sheet in frekammedis.sheet_names:
                    dfrekammedis = frekammedis.parse(sheet)
                    
                    # Cek apakah kolom yang dibutuhkan ada di dalam data
                    self.check_required_columns(dfrekammedis, ['ID PASIEN', 'TANGGAL KUNJUNGAN', 'DIAGNOSA'])
                    
                    list_of_dfs_rekammedis.append(dfrekammedis)

                datarekammedis = pd.concat(list_of_dfs_rekammedis, ignore_index=True).drop_duplicates()
                self.state['datarm'] = datarekammedis
                st.success("Data rekam medis uploaded successfully!")

        except ValueError as e:
            st.error(f"Error in data rekam medis: {e}")
        except (TypeError, IndexError, KeyError) as e:
            st.error(f"Data yang diupload tidak sesuai: {e}")

    def display_data_obat(self):
        if 'dataobat' in self.state:
            if not self.state['dataobat'].empty:
                st.subheader("Data Obat")
                st.write(self.state['dataobat'])
            else:
                st.write("No data available")
        else:
            st.write("No data uploaded yet")

    def display_datarm(self):
        if 'datarm' in self.state:
            if not self.state['datarm'].empty:
                st.subheader("Data Rekam Medis")
                st.write(self.state['datarm'])
            else:
                st.write("No data available")
        else:
            st.write("No data uploaded yet")

    def tampil_dataobat(self):
        if not self.state['dataobat'].empty:
            st.dataframe(self.state['dataobat'])

    def tampil_datarm(self):
        if not self.state['datarm'].empty:
            st.dataframe(self.state['datarm'])

    def menu_data(self):
        self.judul_halaman('Data', 'Import Dataset')
        self.upload_dataobat()
        self.tampil_dataobat()
        self.upload_datarm()
        self.tampil_datarm()


class Preprocessing(Data):

    def preprocess_data(self):
        if not self.state['dataobat'].empty and not self.state['datarm'].empty:
            data_obat = self.state['dataobat'].copy()
            data_rm = self.state['datarm'].copy()

            st.subheader("Data Obat Sebelum Preprocessing")
            st.dataframe(data_obat)
            st.subheader("Data Rekam Medis Sebelum Preprocessing")
            st.dataframe(data_rm)

            # Fill NaN values with 0
            data_obat.fillna(0, inplace=True)
            data_rm.fillna(0, inplace=True)
            st.subheader("Data Obat Setelah Mengisi Nilai NaN dengan 0")
            st.dataframe(data_obat)
            st.subheader("Data Rekam Medis Setelah Mengisi Nilai NaN dengan 0")
            st.dataframe(data_rm)

            # Filter data
            data_obat = data_obat[~data_obat['JENIS'].isin(['Alkes', 'Vaksin'])]
            data_obat = data_obat[data_obat['PEMAKAIAN'] != 0]
            st.subheader("Data Obat Yang Akan Diproses")
            st.dataframe(data_obat)

            # Normalize data
            scaler = MinMaxScaler()
            columns_to_normalize = ['STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR']
            data_obat[columns_to_normalize] = scaler.fit_transform(data_obat[columns_to_normalize])
            st.subheader("Data Obat Setelah Normalisasi")
            st.dataframe(data_obat)

            # Simpan scaler di state
            self.state['scaler'] = scaler

            selected_columns =  ['NO', 'NAMA OBAT', 'JENIS', 'STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR']
            data_obat = data_obat[selected_columns]

            # Merge data
            merged_data = pd.merge(data_obat, data_rm, left_on='NAMA OBAT', right_on='Obat inti', how='inner')
            st.subheader("Data Obat Setelah Penggabungan dengan Data Rekam Medis")
            st.dataframe(merged_data)

            # Add diagnosis columns
            diagnosa_values = merged_data[merged_data['NAMA OBAT'] == merged_data['Obat inti']]['Diagnosa'].unique()
            for diagnosa in diagnosa_values:
                if diagnosa not in data_obat.columns:
                    data_obat[diagnosa] = 0

            for index, row in data_obat.iterrows():
                related_diagnoses = merged_data[(merged_data['NAMA OBAT'] == row['NAMA OBAT']) & (merged_data['Obat inti'] == row['NAMA OBAT'])]['Diagnosa'].unique()
                for diagnosa in related_diagnoses:
                    if diagnosa in data_obat.columns:
                        data_obat.at[index, diagnosa] = 1
            st.subheader("Data Obat Setelah Penambahan Kolom Diagnosa")
            st.dataframe(data_obat)

            if '0' in data_obat.columns:
                data_obat.drop(columns=['0'], inplace=True)

            self.state['processed_data'] = data_obat
            st.success("Preprocessing dan Transformasi data berhasil")
            st.subheader("Data Obat Setelah Preprocessing")
            st.dataframe(data_obat, height=1000)
            st.write(f"Number of rows in dataobat: {data_obat.shape[0]}")

        else:
            st.error("Silahkan upload kedua Data Obat and Data Rekam Medis")

    def menu_preprocessing(self):
        self.judul_halaman('Pre Processing dan Transformation', 'Data Preprocessing')
        self.preprocess_data()

class Dbi(Data):

    def __init__(self):
        super().__init__()

    # Fungsi perhitungan DBI
    def calculate_dbi(self, min_clusters, max_clusters):
        try:
            self.state['results'] = {}
            for i in range(min_clusters, max_clusters + 1):
                hc = AgglomerativeClustering(n_clusters=i, metric='euclidean', linkage='ward')
                y_hc = hc.fit_predict(self.state['processed_data'].select_dtypes(include=[float, int]))  # Hanya pilih kolom numerik
                db_index = davies_bouldin_score(self.state['processed_data'].select_dtypes(include=[float, int]), y_hc)  # Hanya pilih kolom numerik
                self.state['results'][i] = db_index
        except ValueError as e:
            st.error(f"Error during clustering: {e}")

    # Fungsi menampilkan hasil evaluasi DBI
    def show_dbi(self):
        try:
            if self.state['results']:
                self.state['dbi'] = pd.DataFrame(self.state['results'].values(), index=self.state['results'].keys(), columns=['DBI'])
                st.table(self.state['dbi'])
                self.state['dbi'] = self.state['dbi'].round(4)
                min_dbi = self.state['dbi']['DBI'].min()
                optimal_clusters = self.state['dbi']['DBI'].idxmin()
                st.write(f"Nilai terkecil adalah {min_dbi} dengan cluster sebanyak {optimal_clusters}")
                # Simpan informasi cluster optimal ke dalam state
                self.state['optimal_clusters'] = optimal_clusters
                self.state['min_dbi'] = min_dbi
            else:
                st.error("Nilai cluster tidak valid")
        except KeyError as e:
            st.error(f"KeyError: {e}")

    def check_data_input(self):
        # Memeriksa apakah 'processed_data' ada di dalam state
        if 'processed_data' not in self.state or self.state['processed_data'].empty:
            st.warning("Data belum dilakukan proses Pre Processing dan Transformation")
            return False

        return True

    def menu_dbi(self):
        self.judul_halaman('DBI', '')
        if self.check_data_input():
            st.write('Tentukan Rentang Jumlah Cluster')
            cluster_range = st.slider('Rentang Jumlah Cluster', min_value=2, max_value=42, value=(2, 5))

            if st.button('Mulai'):
                min_clusters, max_clusters = cluster_range
                self.calculate_dbi(min_clusters, max_clusters)
                self.show_dbi()



class Clustering(Data):

    def __init__(self):
        super().__init__()

    def reverse_normalization(self, data):
        try:
            if 'scaler' in self.state:
                scaler = self.state['scaler']
                columns_to_reverse = ['STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR']
                
                if all(col in data.columns for col in columns_to_reverse):
                    normalized_data = data[columns_to_reverse]
                    reversed_data = scaler.inverse_transform(normalized_data)
                    reversed_df = pd.DataFrame(reversed_data, columns=columns_to_reverse, index=data.index)
                    return pd.concat([data.drop(columns=columns_to_reverse), reversed_df], axis=1)
                else:
                    st.warning("Kolom data tidak cocok untuk reverse normalization.")
                    return data
            else:
                st.warning("Scaler tidak tersedia untuk reverse normalization.")
                return data
        except Exception as e:
            st.error(f"Error during reverse normalization: {e}")
            return data

    def perform_clustering(self, num_clusters):
        try:
            hc = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
            cluster_labels = hc.fit_predict(self.state['processed_data'].select_dtypes(include=[float, int]))
            self.state['clusters'] = pd.DataFrame(cluster_labels, columns=['Cluster'], index=self.state['processed_data'].index)
            self.state['clusters'] = self.state['processed_data'].join(self.state['clusters'])
        except ValueError as e:
            st.error(f"Error during clustering: {e}")
    
    def show_clusters(self):
        try:
            if not self.state['clusters'].empty:
                # Get unique cluster labels
                cluster_labels = self.state['clusters']['Cluster'].unique()

                # Iterate through each cluster
                for i, cluster_num in enumerate(cluster_labels, start=1):
                    cluster_data = self.state['clusters'][self.state['clusters']['Cluster'] == cluster_num]

                    # Reverse normalization of the data
                    original_cluster_data = self.reverse_normalization(cluster_data)

                    # Add a column for numbering
                    original_cluster_data.insert(0, 'No', range(1, len(original_cluster_data) + 1))

                    # Drop the 'Cluster' column
                    original_cluster_data = original_cluster_data.drop(columns=['Cluster'])

                    # Calculate averages and diagnoses
                    avg_permintaan = original_cluster_data['PENERIMAAN'].mean()
                    avg_pemakaian = original_cluster_data['PEMAKAIAN'].mean()

                    # Exclude columns related to stock and other metrics from diagnoses
                    columns_to_exclude = ['STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR']
                    diagnoses_columns = original_cluster_data.columns[8:]
                    diagnoses_columns = [col for col in diagnoses_columns if col not in columns_to_exclude]
                    valid_columns = [col for col in diagnoses_columns if original_cluster_data[col].sum() > 0]
                    diagnoses = valid_columns

                    # Rearrange columns: No, NAMA OBAT, JENIS, STOK AWAL, PENERIMAAN, PERSEDIAAN, PEMAKAIAN, STOK AKHIR, Diagnoses...
                    columns_order = ['No', 'NAMA OBAT', 'JENIS', 'STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR'] + diagnoses
                    original_cluster_data = original_cluster_data[columns_order]

                    # Display information for the cluster
                    st.write(f"### Cluster {i}")
                    st.write(f"Pada cluster ini, terdapat {len(cluster_data)} jenis obat.")
                    st.write(f"Rata-rata permintaan: {avg_permintaan:.2f}")
                    st.write(f"Rata-rata pemakaian: {avg_pemakaian:.2f}")
                    st.write(f"Penyakit yang dapat disembuhkan: {', '.join(diagnoses) if diagnoses else 'Tidak ada diagnosa yang cocok'}")
                    st.dataframe(original_cluster_data)
            else:
                st.warning("Belum ada cluster yang terbentuk.")
        except KeyError as e:
            st.error(f"KeyError: {e}")

    def check_data_input(self):
        # Memeriksa apakah 'processed_data' ada di dalam state
        if 'processed_data' not in self.state or self.state['processed_data'].empty:
            st.warning("Data belum dilakukan proses Pre Processing dan Transformation")
            return False

        return True

    def menu_clustering(self):
        self.judul_halaman('Clustering', '')
        if self.check_data_input():
            if 'optimal_clusters' in self.state and 'min_dbi' in self.state:
                st.write(f"Menurut hasil perhitungan DBI, cluster terbaik yang bisa dibentuk adalah cluster sebanyak {self.state['optimal_clusters']} dengan nilai DBI terkecil {self.state['min_dbi']}.")
            
            st.write('Tentukan Jumlah Cluster')
            num_clusters = st.number_input('Jumlah Cluster', value=self.state.get('optimal_clusters', 2), min_value=2, step=1, key='num_clusters')

            if st.button('Mulai Clustering'):
                self.perform_clustering(num_clusters)
                self.show_clusters()


class Rekomendasi(Data):

    def __init__(self):
        super().__init__()

    def reverse_normalization(self, data):
        try:
            if 'scaler' in self.state:
                scaler = self.state['scaler']
                columns_to_reverse = ['STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR']
                
                if all(col in data.columns for col in columns_to_reverse):
                    normalized_data = data[columns_to_reverse]
                    reversed_data = scaler.inverse_transform(normalized_data)
                    reversed_df = pd.DataFrame(reversed_data, columns=columns_to_reverse, index=data.index)
                    
                    return pd.concat([data.drop(columns=columns_to_reverse), reversed_df], axis=1)
                else:
                    st.warning("Kolom data tidak cocok untuk reverse normalization.")
                    return data
            else:
                st.warning("Scaler tidak tersedia untuk reverse normalization.")
                return data
        except Exception as e:
            st.error(f"Error during reverse normalization: {e}")
            return data

    def calculate_recommendation(self, selected_cluster):
        try:
            if not self.state['clusters'].empty:
                cluster_data = self.state['clusters'][self.state['clusters']['Cluster'] == selected_cluster]
                
                columns_to_reverse = ['STOK AWAL', 'PENERIMAAN', 'PERSEDIAAN', 'PEMAKAIAN', 'STOK AKHIR']
                cluster_data_reversed = self.reverse_normalization(cluster_data)

                # Menghitung rekomendasi permintaan untuk bulan depan
                cluster_data_reversed['Rekomendasi Permintaan'] = cluster_data_reversed.apply(
                    lambda row: np.ceil(row['PENERIMAAN'] * 1.10) if row['PENERIMAAN'] > 0 else ('1 BOX'),
                    axis=1
                )
                
                # Menampilkan data sesuai urutan: PENERIMAAN, PEMAKAIAN, STOK AKHIR, REKOMENDASI
                result_data = cluster_data_reversed[['STOK AWAL','PENERIMAAN', 'PEMAKAIAN', 'STOK AKHIR', 'Rekomendasi Permintaan']]
                
                st.success(f"Rekomendasi permintaan obat untuk bulan Juni pada Cluster {selected_cluster + 1} telah dihitung dan data telah dibalik ke bentuk asli.")
                st.write(f"### Data Obat pada Cluster {selected_cluster + 1}")
                st.dataframe(result_data)
            else:
                st.warning("Belum ada cluster yang terbentuk.")
        except KeyError as e:
            st.error(f"KeyError: {e}")

    def menu_rekomendasi(self):
        self.judul_halaman('Rekomendasi Permintaan Obat', '')

        if not self.state.get('clusters', pd.DataFrame()).empty:
            st.write('Pilih Cluster untuk Rekomendasi Permintaan')
            
            # Pilihan cluster dimulai dari 1
            cluster_options = sorted(self.state['clusters']['Cluster'].unique())
            cluster_options = [i+1 for i in range(len(cluster_options))]  # Memulai opsi cluster dari 1
            
            selected_cluster = st.selectbox('Pilih Cluster', options=cluster_options, key='selected_cluster')

            if st.button('Hitung Rekomendasi Permintaan'):
                # Sesuaikan indeks cluster karena telah diubah mulai dari 1
                actual_cluster = selected_cluster - 1
                self.calculate_recommendation(actual_cluster)
        else:
            st.warning("Data belum diolah menjadi cluster. Silakan lakukan proses clustering terlebih dahulu.")



if __name__ == "__main__":
    # Create an instance of the main class
    main = MainClass()
    
    main.sidebar_menu()
